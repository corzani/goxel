#version 330

/* Goxel 3D voxels editor
 *
 * copyright (c) 2026 Guillaume Chereau <guillaume@noctua-software.com>
 *
 * Goxel is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.

 * Goxel is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.

 * You should have received a copy of the GNU General Public License along with
 * goxel.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * GPU path tracer (see src/pathtracer_gpu.c).
 *
 * The rays are traced directly into the voxels, stored as a two-level grid:
 *
 *   u_table - One texel per tile, with the position of the tile in the
 *             atlas, or zero if the tile is empty.
 *   u_atlas - The voxels of all the non empty tiles: sRGB color, and the
 *             material index + 1 in the alpha channel (0 for no voxel).
 *
 * The ACCUMULATE pass adds one sample per pixel into the accumulation
 * buffer.  The DISPLAY pass averages the samples, applies the tone mapping
 * and outputs sRGB colors.
 */

#define TILE_SIZE 16
#define MAX_STEPS 1024
#define FLOOR_MATERIAL 255
#define PI 3.14159265359

#ifdef VERTEX_SHADER

in vec2 a_pos;

void main()
{
    gl_Position = vec4(a_pos, 0.0, 1.0);
}

#endif // VERTEX_SHADER

#ifdef FRAGMENT_SHADER

layout(location = 0) out vec4 out_color;

#ifdef DISPLAY

uniform sampler2D u_accum;
uniform float     u_samples;
uniform float     u_exposure;

// ACES filmic tone mapping curve (fit by Krzysztof Narkowicz).
vec3 aces(vec3 x)
{
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 c)
{
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055,
               step(vec3(0.0031308), c));
}

void main()
{
    ivec2 size = textureSize(u_accum, 0);
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec4 v;

    // Flip the image, so that the first row is the top of the image.
    p.y = size.y - 1 - p.y;
    v = texelFetch(u_accum, p, 0) / max(u_samples, 1.0);
    // The accumulated colors are premultiplied by the alpha.
    if (v.a > 0.0) v.rgb /= v.a;
    out_color = vec4(linear_to_srgb(aces(v.rgb * u_exposure)), v.a);
}

#endif // DISPLAY

#ifdef ACCUMULATE

uniform sampler2D  u_prev;
uniform usampler3D u_table;
uniform sampler3D  u_atlas;
uniform sampler2D  u_materials;

uniform ivec3 u_grid_origin;  // Position of the grid (in voxels).
uniform ivec3 u_grid_size;    // Size of the grid (in tiles).

uniform int   u_seed;
uniform int   u_bounces;
uniform mat4  u_camera;       // Camera transformation (looking toward -z).
uniform float u_tan_fovy;     // Tangent of the half vertical field of view.
uniform float u_ortho_size;   // Half width of the view if ortho, else zero.

uniform vec3  u_sun_dir;      // Direction toward the sun.
uniform float u_sun_intensity;
uniform float u_sun_cos;      // Cosine of the sun angular radius.

uniform int   u_world_type;
uniform vec3  u_world_color;  // Multiplied by the world energy.
uniform float u_world_energy;

uniform int   u_floor;        // Set to 1 if there is a floor.
uniform float u_floor_z;
uniform vec4  u_floor_rect;   // [min_x, min_y, max_x, max_y].

struct hit_t {
    float t;
    vec3  pos;
    vec3  normal;
    vec3  color;
    int   material;
};

struct material_t {
    vec3  albedo;
    float opacity;
    vec3  emission;
    float metallic;
    float roughness;
};

uint g_rng_state;

// PCG hash based random generator, returns a value in [0, 1).
float rand()
{
    uint w;
    g_rng_state = g_rng_state * 747796405u + 2891336453u;
    w = ((g_rng_state >> ((g_rng_state >> 28u) + 4u)) ^ g_rng_state) *
        277803737u;
    w = (w >> 22u) ^ w;
    return float(w >> 8u) / 16777216.0;
}

float luminance(vec3 c)
{
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

int argmin(vec3 v)
{
    return (v.x < v.y) ? ((v.x < v.z) ? 0 : 2) : ((v.y < v.z) ? 1 : 2);
}

// Orthonormal basis around a normal (Duff et al. 2017).
mat3 get_basis(vec3 n)
{
    float s = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float b = n.x * n.y * a;
    return mat3(vec3(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
                vec3(b, s + n.y * n.y * a, -n.y),
                n);
}

material_t get_material(int index, vec3 color)
{
    material_t m;
    vec4 v0 = texelFetch(u_materials, ivec2(index, 0), 0);
    vec4 v1 = texelFetch(u_materials, ivec2(index, 1), 0);
    vec4 v2 = texelFetch(u_materials, ivec2(index, 2), 0);
    m.albedo = v0.rgb * color;
    m.opacity = v0.a;
    m.emission = v1.rgb * color;
    m.metallic = v2.r;
    m.roughness = v2.g;
    return m;
}

/*
 * Trace a ray into the voxels, using a DDA traversal that skips the empty
 * tiles.  Return true if we hit a voxel before tmax.
 */
bool trace_voxels(vec3 ro, vec3 rd, float tmax, inout hit_t hit)
{
    vec3 o, d, inv, t0, t1, near, far, tnext, tb, gmax;
    ivec3 step_, cell, tile, block;
    float t, texit, opacity;
    int i, axis;
    uint v;
    vec4 voxel;

    if (u_grid_size.x == 0) return false;

    // Work in grid coordinates.
    o = ro - vec3(u_grid_origin);
    // Avoid divisions by zero.
    d = mix(rd, vec3(1e-9), lessThan(abs(rd), vec3(1e-9)));
    inv = 1.0 / d;
    gmax = vec3(u_grid_size * TILE_SIZE);

    // Clip the ray to the grid box.
    t0 = -o * inv;
    t1 = (gmax - o) * inv;
    near = min(t0, t1);
    far = max(t0, t1);
    t = max(max(near.x, near.y), near.z);
    texit = min(min(far.x, far.y), min(far.z, tmax));
    if (t < 0.0) {
        t = 0.0;
        axis = -1; // Ray starting inside the grid.
    } else {
        axis = argmin(-near);
    }
    if (t >= texit) return false;

    step_ = ivec3(sign(d));
    cell = clamp(ivec3(floor(o + d * t)), ivec3(0), ivec3(gmax) - 1);
    tnext = (vec3(cell) + max(vec3(step_), 0.0) - o) * inv;

    for (i = 0; i < MAX_STEPS; i++) {
        if (any(lessThan(cell, ivec3(0))) ||
            any(greaterThanEqual(cell, ivec3(gmax))))
            return false;
        tile = cell / TILE_SIZE;
        v = texelFetch(u_table, tile, 0).r;

        if (v == 0u) {
            // Empty tile: jump directly to the next one.
            tb = (vec3((tile + max(step_, 0)) * TILE_SIZE) - o) * inv;
            axis = argmin(tb);
            t = tb[axis];
            if (t >= texit) return false;
            cell = ivec3(floor(o + d * t));
            cell[axis] = (step_[axis] > 0) ? (tile[axis] + 1) * TILE_SIZE
                                           : tile[axis] * TILE_SIZE - 1;
            tnext = (vec3(cell) + max(vec3(step_), 0.0) - o) * inv;
            continue;
        }

        block = ivec3(v & 1023u, (v >> 10u) & 1023u, (v >> 20u) & 1023u);
        voxel = texelFetch(u_atlas, (block - tile) * TILE_SIZE + cell, 0);
        if (voxel.a > 0.0) {
            hit.material = int(voxel.a * 255.0 + 0.5) - 1;
            opacity = texelFetch(u_materials, ivec2(hit.material, 0), 0).a;
            // Semi transparent materials are stochastically skipped.
            if (opacity > rand()) {
                hit.t = t;
                hit.color = voxel.rgb;
                if (axis == -1) {
                    hit.normal = -rd;
                    hit.pos = ro + rd * t;
                } else {
                    hit.normal = vec3(0.0);
                    hit.normal[axis] = -float(step_[axis]);
                    // Snap the position to the voxel face.
                    hit.pos = o + d * t;
                    hit.pos[axis] = float(cell[axis] +
                                          ((step_[axis] > 0) ? 0 : 1));
                    hit.pos += vec3(u_grid_origin);
                }
                return true;
            }
        }

        axis = argmin(tnext);
        t = tnext[axis];
        if (t >= texit) return false;
        cell[axis] += step_[axis];
        tnext[axis] += abs(inv[axis]);
    }
    return false;
}

bool trace_floor(vec3 ro, vec3 rd, float tmax, inout hit_t hit)
{
    float t;
    vec3 p;

    if (u_floor == 0 || abs(rd.z) < 1e-9) return false;
    t = (u_floor_z - ro.z) / rd.z;
    if (t <= 0.0 || t >= tmax) return false;
    p = ro + rd * t;
    if (any(lessThan(p.xy, u_floor_rect.xy)) ||
        any(greaterThan(p.xy, u_floor_rect.zw)))
        return false;
    hit.t = t;
    hit.pos = vec3(p.xy, u_floor_z);
    hit.normal = vec3(0.0, 0.0, (rd.z < 0.0) ? 1.0 : -1.0);
    hit.color = vec3(1.0);
    hit.material = FLOOR_MATERIAL;
    return true;
}

bool trace(vec3 ro, vec3 rd, out hit_t hit)
{
    bool ret = trace_voxels(ro, rd, 1e30, hit);
    return trace_floor(ro, rd, ret ? hit.t : 1e30, hit) || ret;
}

vec3 get_sky(vec3 d)
{
    const vec3 ZENITH = vec3(0.20, 0.40, 0.85);
    const vec3 HORIZON = vec3(0.80, 0.86, 0.92);

    if (u_world_type == 0) return vec3(0.0); // None.
    if (u_world_type == 1) return u_world_color; // Uniform.
    // Sky: blue gradient above the horizon.  Like with the CPU renderer, the
    // world color is used for the ground.
    if (d.z >= 0.0) return mix(HORIZON, ZENITH, sqrt(d.z)) * u_world_energy;
    return mix(HORIZON * u_world_energy, u_world_color * HORIZON,
               smoothstep(0.0, 0.8, -d.z));
}

// Clamp the indirect light contributions, to limit the fireflies.
vec3 clamp_indirect(vec3 c, int bounce)
{
    float l = luminance(c);
    if (bounce == 0 || l <= 8.0) return c;
    return c * (8.0 / l);
}

vec3 sample_cone(vec3 dir, float cos_max)
{
    float cos_t = mix(1.0, cos_max, rand());
    float sin_t = sqrt(max(0.0, 1.0 - cos_t * cos_t));
    float phi = 2.0 * PI * rand();
    return get_basis(dir) * vec3(cos(phi) * sin_t, sin(phi) * sin_t, cos_t);
}

vec3 fresnel(vec3 f0, float cos_t)
{
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cos_t, 0.0, 1.0), 5.0);
}

float get_alpha(float roughness)
{
    return max(roughness * roughness, 0.005);
}

float ggx_d(float n_dot_h, float a)
{
    float a2 = a * a;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * d * d);
}

float ggx_g1(float n_dot_x, float a)
{
    float a2 = a * a;
    return 2.0 * n_dot_x /
           (n_dot_x + sqrt(a2 + (1.0 - a2) * n_dot_x * n_dot_x));
}

// Return the BRDF multiplied by the cosine term.
vec3 eval_brdf(material_t m, vec3 n, vec3 wo, vec3 wi)
{
    float n_dot_l = dot(n, wi);
    float n_dot_v = dot(n, wo);
    float a = get_alpha(m.roughness);
    vec3 h, f0, f, spec, diff;

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) return vec3(0.0);
    h = normalize(wi + wo);
    f0 = mix(vec3(0.04), m.albedo, m.metallic);
    f = fresnel(f0, dot(wo, h));
    spec = f * ggx_d(min(dot(n, h), 1.0), a) *
           ggx_g1(n_dot_l, a) * ggx_g1(n_dot_v, a) / (4.0 * n_dot_l * n_dot_v);
    diff = (1.0 - m.metallic) * (1.0 - f) * m.albedo / PI;
    return (diff + spec) * n_dot_l;
}

// Sample the GGX distribution of visible normals (Heitz 2018).
vec3 sample_ggx_vndf(vec3 v, float a, float u1, float u2)
{
    vec3 vh = normalize(vec3(a * v.x, a * v.y, v.z));
    float lensq = vh.x * vh.x + vh.y * vh.y;
    vec3 b1 = (lensq > 0.0) ? vec3(-vh.y, vh.x, 0.0) * inversesqrt(lensq)
                            : vec3(1.0, 0.0, 0.0);
    vec3 b2 = cross(vh, b1);
    float r = sqrt(u1);
    float phi = 2.0 * PI * u2;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi);
    float s = 0.5 * (1.0 + vh.z);
    vec3 nh;
    p2 = (1.0 - s) * sqrt(1.0 - p1 * p1) + s * p2;
    nh = p1 * b1 + p2 * b2 + sqrt(max(0.0, 1.0 - p1 * p1 - p2 * p2)) * vh;
    return normalize(vec3(a * nh.x, a * nh.y, max(0.0, nh.z)));
}

/*
 * Sample a new direction from the BRDF, choosing randomly between the
 * diffuse and specular lobes.  Return the BRDF multiplied by the cosine
 * term, divided by the pdf.
 */
vec3 sample_brdf(material_t m, vec3 n, vec3 wo, out vec3 wi,
                 out bool specular)
{
    float a = get_alpha(m.roughness);
    vec3 f0 = mix(vec3(0.04), m.albedo, m.metallic);
    vec3 f = fresnel(f0, dot(n, wo));
    float spec_w = luminance(f);
    float diff_w = (1.0 - m.metallic) * luminance(m.albedo) * (1.0 - spec_w);
    float p_spec = 1.0;
    mat3 basis = get_basis(n);
    vec3 h;

    if (diff_w > 0.0 && spec_w > 0.0)
        p_spec = clamp(spec_w / (spec_w + diff_w), 0.1, 0.9);
    else if (diff_w > 0.0)
        p_spec = 0.0;

    specular = rand() < p_spec;
    if (specular) {
        h = basis * sample_ggx_vndf(transpose(basis) * wo, a, rand(), rand());
        wi = reflect(-wo, h);
        if (dot(n, wi) <= 0.0) return vec3(0.0);
        return fresnel(f0, dot(wo, h)) * ggx_g1(dot(n, wi), a) / p_spec;
    } else {
        float u1 = rand(), u2 = rand();
        float r = sqrt(u1), phi = 2.0 * PI * u2;
        wi = basis * vec3(r * cos(phi), r * sin(phi), sqrt(1.0 - u1));
        f = fresnel(f0, dot(wo, normalize(wi + wo)));
        return (1.0 - m.metallic) * (1.0 - f) * m.albedo / (1.0 - p_spec);
    }
}

void main()
{
    ivec2 size = textureSize(u_prev, 0);
    ivec2 px = ivec2(gl_FragCoord.xy);
    float aspect = float(size.x) / float(size.y);
    float alpha = 1.0, q, path_roughness = 0.0;
    vec2 ndc;
    vec3 ro, rd, wo, wi, w, l, pos;
    vec3 radiance = vec3(0.0), throughput = vec3(1.0);
    hit_t hit, shadow_hit;
    material_t m;
    int bounce;
    bool specular;

    g_rng_state = (uint(px.x) * 73856093u) ^ (uint(px.y) * 19349663u) ^
                  (uint(u_seed) * 83492791u);
    rand();

    // Camera ray, with some jitter for the anti aliasing.
    ndc = (vec2(px) + vec2(rand(), rand())) / vec2(size) * 2.0 - 1.0;
    if (u_ortho_size > 0.0) {
        rd = -normalize(u_camera[2].xyz);
        ro = u_camera[3].xyz - rd * 1024.0 +
             normalize(u_camera[0].xyz) * ndc.x * u_ortho_size +
             normalize(u_camera[1].xyz) * ndc.y * u_ortho_size / aspect;
    } else {
        ro = u_camera[3].xyz;
        rd = normalize(-normalize(u_camera[2].xyz) +
            normalize(u_camera[0].xyz) * ndc.x * u_tan_fovy * aspect +
            normalize(u_camera[1].xyz) * ndc.y * u_tan_fovy);
    }

    for (bounce = 0; bounce <= u_bounces; bounce++) {
        if (!trace(ro, rd, hit)) {
            if (bounce == 0 && u_world_type == 0) alpha = 0.0;
            radiance += clamp_indirect(throughput * get_sky(rd), bounce);
            break;
        }
        m = get_material(hit.material, hit.color);
        // Path regularization: the surfaces seen after a rough bounce are
        // made at least as rough, this removes most of the caustics noise.
        m.roughness = max(m.roughness, path_roughness);
        wo = -rd;
        pos = hit.pos + hit.normal * 1e-3;
        radiance += clamp_indirect(throughput * m.emission, bounce);

        // Direct light from the sun.
        l = sample_cone(u_sun_dir, u_sun_cos);
        if (u_sun_intensity > 0.0 && dot(l, hit.normal) > 0.0 &&
            !trace(pos, l, shadow_hit)) {
            radiance += clamp_indirect(
                    throughput * eval_brdf(m, hit.normal, wo, l) *
                    u_sun_intensity, bounce);
        }

        if (bounce == u_bounces) break;
        w = sample_brdf(m, hit.normal, wo, wi, specular);
        if (max(max(w.r, w.g), w.b) <= 0.0) break;
        throughput *= w;
        path_roughness = max(path_roughness, specular ? m.roughness : 1.0);

        // Russian roulette.
        if (bounce >= 2) {
            q = clamp(max(max(throughput.r, throughput.g), throughput.b),
                      0.05, 0.95);
            if (rand() > q) break;
            throughput /= q;
        }
        ro = pos;
        rd = wi;
    }

    // Clamp the fireflies.
    q = luminance(radiance);
    if (q > 32.0) radiance *= 32.0 / q;

    out_color = texelFetch(u_prev, px, 0) + vec4(radiance, alpha);
}

#endif // ACCUMULATE

#endif // FRAGMENT_SHADER
