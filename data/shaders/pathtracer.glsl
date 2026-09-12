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
 * buffer.  The BLOOM_DOWN and BLOOM_UP passes compute the bloom, and the
 * DISPLAY pass averages the samples, applies the tone mapping and outputs
 * sRGB colors.
 */

#define TILE_SIZE 16
#define MAX_STEPS 1024
#define LIGHTS_TEX_WIDTH 1024 // Must match src/pathtracer_gpu.c
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
uniform sampler2D u_bloom;
uniform float     u_bloom_intensity;

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
    v.rgb += texture(u_bloom, (vec2(p) + 0.5) / vec2(size)).rgb *
             u_bloom_intensity;
    // The accumulated colors are premultiplied by the alpha.
    if (v.a > 0.0) v.rgb /= v.a;
    out_color = vec4(linear_to_srgb(aces(v.rgb * u_exposure)), v.a);
}

#endif // DISPLAY

#ifdef BLOOM_DOWN

// Bloom downsample pass, with a soft threshold for the first level.
uniform sampler2D u_src;
uniform vec2      u_dst_size;
uniform float     u_scale;      // Multiplier of the source colors.
uniform float     u_threshold;  // Only used if > 0.

void main()
{
    vec2 uv = gl_FragCoord.xy / u_dst_size;
    vec2 t = 1.0 / vec2(textureSize(u_src, 0));
    vec3 c;
    float b, soft;

    c = (texture(u_src, uv + t * vec2(-1.0, -1.0)).rgb +
         texture(u_src, uv + t * vec2(+1.0, -1.0)).rgb +
         texture(u_src, uv + t * vec2(-1.0, +1.0)).rgb +
         texture(u_src, uv + t * vec2(+1.0, +1.0)).rgb) * 0.25 * u_scale;
    if (u_threshold > 0.0) {
        b = max(c.r, max(c.g, c.b));
        soft = clamp(b - u_threshold + 0.5, 0.0, 1.0);
        soft = soft * soft * 0.5;
        c *= max(soft, b - u_threshold) / max(b, 1e-4);
    }
    out_color = vec4(c, 1.0);
}

#endif // BLOOM_DOWN

#ifdef BLOOM_UP

// Bloom upsample pass: add the blurred lower level to this level.
uniform sampler2D u_src;    // Lower resolution level.
uniform sampler2D u_base;   // Downsampled image at this level.
uniform vec2      u_dst_size;

void main()
{
    vec2 uv = gl_FragCoord.xy / u_dst_size;
    vec2 t = 1.0 / vec2(textureSize(u_src, 0));
    vec3 c;

    // 3x3 tent filter.
    c = texture(u_src, uv).rgb * 4.0;
    c += (texture(u_src, uv + vec2(t.x, 0.0)).rgb +
          texture(u_src, uv - vec2(t.x, 0.0)).rgb +
          texture(u_src, uv + vec2(0.0, t.y)).rgb +
          texture(u_src, uv - vec2(0.0, t.y)).rgb) * 2.0;
    c += texture(u_src, uv + t).rgb + texture(u_src, uv - t).rgb +
         texture(u_src, uv + vec2(t.x, -t.y)).rgb +
         texture(u_src, uv + vec2(-t.x, t.y)).rgb;
    out_color = vec4(texture(u_base, uv).rgb + c / 16.0, 1.0);
}

#endif // BLOOM_UP

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
uniform float u_aperture;     // Lens diameter, zero for no depth of field.
uniform float u_focus;        // Focus distance.

uniform vec3  u_sun_dir;      // Direction toward the sun.
uniform float u_sun_intensity;
uniform float u_sun_cos;      // Cosine of the sun angular radius.

uniform int   u_world_type;
uniform vec3  u_world_color;  // Multiplied by the world energy.
uniform float u_world_energy;
uniform vec3  u_sky_zenith;   // Procedural sky colors.
uniform vec3  u_sky_horizon;
uniform vec3  u_sky_ground;

uniform sampler2D u_env;        // Environment image (equirectangular).
uniform sampler2D u_env_cond;   // Cumulated luminance of each row.
uniform sampler2D u_env_marg;   // Cumulated luminance of the rows.
uniform ivec2 u_env_size;
uniform float u_env_integral;   // Sum of the weighted luminances.

uniform int   u_floor;        // Set to 1 if there is a floor.
uniform float u_floor_z;
uniform vec4  u_floor_rect;   // [min_x, min_y, max_x, max_y].

uniform sampler2D u_lights_pos;   // Emissive voxels position, and CDF.
uniform sampler2D u_lights_emit;  // Emissive voxels emission.
uniform int   u_lights_count;
uniform float u_lights_power;     // Sum of the lights luminance.

struct hit_t {
    float t;
    vec3  pos;
    vec3  normal;       // Facing the incoming ray.
    vec3  color;
    int   material;     // -1 when leaving a refractive material.
};

struct material_t {
    vec3  albedo;
    float opacity;
    vec3  emission;
    float metallic;
    float roughness;
    float ior;          // Zero for non refractive materials.
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

float max3(vec3 v)
{
    return max(max(v.x, v.y), v.z);
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
    m.emission = v1.rgb; // Absolute color, like the other renderers.
    m.metallic = v2.r;
    m.roughness = v2.g;
    m.ior = v2.b;
    return m;
}

float get_ior(int material)
{
    return texelFetch(u_materials, ivec2(material, 2), 0).b;
}

void set_hit(inout hit_t hit, float t, int axis, ivec3 step_, ivec3 cell,
             vec3 ro, vec3 rd, vec3 o, vec3 d, vec4 voxel, int material)
{
    hit.t = t;
    hit.material = material;
    hit.color = (material >= 0) ? voxel.rgb : vec3(1.0);
    if (axis == -1) {
        hit.normal = -rd;
        hit.pos = ro + rd * t;
    } else {
        hit.normal = vec3(0.0);
        hit.normal[axis] = -float(step_[axis]);
        // Snap the position to the voxel face.
        hit.pos = o + d * t;
        hit.pos[axis] = float(cell[axis] + ((step_[axis] > 0) ? 0 : 1));
        hit.pos += vec3(u_grid_origin);
    }
}

/*
 * Trace a ray into the voxels, using a DDA traversal that skips the empty
 * tiles.  Return true if we hit something before tmax.
 *
 * medium - Index of the refractive material the ray is inside of, or -1.
 *          Inside a material, we stop at the first voxel with a different
 *          material (hit.material is set to -1 for the empty voxels).
 * shadow - For the shadow rays: the refractive voxels don't block the ray,
 *          but tint the transmittance 'tr' when entering them.
 */
bool trace_voxels(vec3 ro, vec3 rd, float tmax, int medium, bool shadow,
                  inout vec3 tr, inout hit_t hit)
{
    vec3 o, d, inv, t0, t1, near, far, tnext, tb, gmax;
    ivec3 step_, cell, tile, block;
    float t, texit;
    int i, axis, mat, last = -1;
    uint v;
    vec4 voxel = vec4(0.0);

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
            if (medium >= 0) { // Leaving the refractive material.
                set_hit(hit, t, axis, step_, cell, ro, rd, o, d, voxel, -1);
                return true;
            }
            // Empty tile: jump directly to the next one.
            tb = (vec3((tile + max(step_, 0)) * TILE_SIZE) - o) * inv;
            axis = argmin(tb);
            t = tb[axis];
            if (t >= texit) return false;
            cell = ivec3(floor(o + d * t));
            cell[axis] = (step_[axis] > 0) ? (tile[axis] + 1) * TILE_SIZE
                                           : tile[axis] * TILE_SIZE - 1;
            tnext = (vec3(cell) + max(vec3(step_), 0.0) - o) * inv;
            last = -1;
            continue;
        }

        block = ivec3(v & 1023u, (v >> 10u) & 1023u, (v >> 20u) & 1023u);
        voxel = texelFetch(u_atlas, (block - tile) * TILE_SIZE + cell, 0);
        mat = (voxel.a > 0.0) ? int(voxel.a * 255.0 + 0.5) - 1 : -1;

        if (medium >= 0) {
            if (mat != medium) {
                set_hit(hit, t, axis, step_, cell, ro, rd, o, d, voxel, mat);
                return true;
            }
        } else if (mat >= 0) {
            if (get_ior(mat) > 0.0) {
                if (!shadow) {
                    set_hit(hit, t, axis, step_, cell, ro, rd, o, d, voxel,
                            mat);
                    return true;
                }
                if (mat != last) {
                    tr *= voxel.rgb *
                          texelFetch(u_materials, ivec2(mat, 0), 0).rgb;
                }
            } else if (texelFetch(u_materials, ivec2(mat, 0), 0).a > rand()) {
                // Semi transparent materials are stochastically skipped.
                set_hit(hit, t, axis, step_, cell, ro, rd, o, d, voxel, mat);
                return true;
            }
        }
        last = mat;

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

bool trace(vec3 ro, vec3 rd, int medium, out hit_t hit)
{
    vec3 tr = vec3(1.0);
    bool ret = trace_voxels(ro, rd, 1e30, medium, false, tr, hit);
    return trace_floor(ro, rd, ret ? hit.t : 1e30, hit) || ret;
}

// Light transmitted along a shadow ray, zero if it is blocked.
vec3 transmittance(vec3 ro, vec3 rd, float tmax)
{
    hit_t hit;
    vec3 tr = vec3(1.0);

    if (trace_voxels(ro, rd, tmax, -1, true, tr, hit) ||
        trace_floor(ro, rd, tmax, hit))
        return vec3(0.0);
    return tr;
}

vec3 get_sky(vec3 d)
{
    float theta, phi;

    if (u_world_type == 0) return vec3(0.0); // None.
    if (u_world_type == 1) return u_world_color; // Uniform.
    if (u_world_type == 3) { // Image.
        theta = acos(clamp(d.z, -1.0, 1.0));
        phi = atan(d.y, d.x);
        return texture(u_env, vec2(phi / (2.0 * PI) + 0.5, theta / PI)).rgb *
               u_world_energy;
    }
    // Procedural sky: gradient from the ground to the horizon and the zenith.
    if (d.z >= 0.0)
        return mix(u_sky_horizon, u_sky_zenith, sqrt(d.z)) * u_world_energy;
    return mix(u_sky_horizon, u_sky_ground, smoothstep(0.0, 0.4, -d.z)) *
           u_world_energy;
}

// Clamp the indirect light contributions, to limit the fireflies.
vec3 clamp_indirect(vec3 c, int bounce)
{
    float l = luminance(c);
    if (bounce == 0 || l <= 8.0) return c;
    return c * (8.0 / l);
}

// Power heuristic for multiple importance sampling.
float mis(float a, float b)
{
    a *= a;
    b *= b;
    return (a + b > 0.0) ? a / (a + b) : 0.0;
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

// Probability to sample the specular lobe instead of the diffuse one.
float get_spec_prob(material_t m, vec3 n, vec3 wo)
{
    vec3 f0 = mix(vec3(0.04), m.albedo, m.metallic);
    float spec_w = luminance(fresnel(f0, dot(n, wo)));
    float diff_w = (1.0 - m.metallic) * luminance(m.albedo) * (1.0 - spec_w);

    if (diff_w > 0.0 && spec_w > 0.0)
        return clamp(spec_w / (spec_w + diff_w), 0.1, 0.9);
    return (diff_w > 0.0) ? 0.0 : 1.0;
}

// Solid angle pdf of sampling a given direction with sample_brdf.
float brdf_pdf(material_t m, vec3 n, vec3 wo, vec3 wi)
{
    float n_dot_l = dot(n, wi);
    float n_dot_v = dot(n, wo);
    float a = get_alpha(m.roughness);
    float p_spec, spec;
    vec3 h;

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) return 0.0;
    p_spec = get_spec_prob(m, n, wo);
    h = normalize(wi + wo);
    spec = ggx_g1(n_dot_v, a) * ggx_d(min(dot(n, h), 1.0), a) /
           (4.0 * n_dot_v);
    return p_spec * spec + (1.0 - p_spec) * n_dot_l / PI;
}

/*
 * Sample a new direction from the BRDF, choosing randomly between the
 * diffuse and specular lobes.  Return the BRDF multiplied by the cosine
 * term, divided by the pdf.
 */
vec3 sample_brdf(material_t m, vec3 n, vec3 wo, out vec3 wi,
                 out bool specular, out float pdf)
{
    float a = get_alpha(m.roughness);
    vec3 f0 = mix(vec3(0.04), m.albedo, m.metallic);
    float p_spec = get_spec_prob(m, n, wo);
    mat3 basis = get_basis(n);
    vec3 h, f;

    pdf = 0.0;
    specular = rand() < p_spec;
    if (specular) {
        h = basis * sample_ggx_vndf(transpose(basis) * wo, a, rand(), rand());
        wi = reflect(-wo, h);
        if (dot(n, wi) <= 0.0) return vec3(0.0);
        pdf = brdf_pdf(m, n, wo, wi);
        return fresnel(f0, dot(wo, h)) * ggx_g1(dot(n, wi), a) / p_spec;
    } else {
        float u1 = rand(), u2 = rand();
        float r = sqrt(u1), phi = 2.0 * PI * u2;
        wi = basis * vec3(r * cos(phi), r * sin(phi), sqrt(1.0 - u1));
        pdf = brdf_pdf(m, n, wo, wi);
        f = fresnel(f0, dot(wo, normalize(wi + wo)));
        return (1.0 - m.metallic) * (1.0 - f) * m.albedo / (1.0 - p_spec);
    }
}

// Fresnel reflectance of a dielectric interface, with eta = eta_i / eta_t.
float fresnel_dielectric(float cos_i, float eta)
{
    float sin2_t = eta * eta * (1.0 - cos_i * cos_i);
    float cos_t, rs, rp;

    if (sin2_t >= 1.0) return 1.0;
    cos_t = sqrt(1.0 - sin2_t);
    rs = (eta * cos_i - cos_t) / (eta * cos_i + cos_t);
    rp = (cos_i - eta * cos_t) / (cos_i + eta * cos_t);
    return 0.5 * (rs * rs + rp * rp);
}

/*
 * Sample the reflection or the refraction on a dielectric interface, with
 * n facing the incoming ray.  Return true if the ray is refracted.
 */
bool sample_dielectric(vec3 rd, vec3 n, float eta, float roughness,
                       out vec3 wi)
{
    vec3 h = n;
    mat3 basis;

    if (roughness > 0.0) {
        basis = get_basis(n);
        h = basis * sample_ggx_vndf(transpose(basis) * -rd,
                                    get_alpha(roughness), rand(), rand());
    }
    if (rand() < fresnel_dielectric(clamp(dot(-rd, h), 0.0, 1.0), eta)) {
        wi = reflect(rd, h);
        return false;
    }
    wi = refract(rd, h, eta);
    if (dot(wi, wi) == 0.0) { // Total internal reflection.
        wi = reflect(rd, h);
        return false;
    }
    return true;
}

vec4 get_light(sampler2D tex, int i)
{
    return texelFetch(tex, ivec2(i % LIGHTS_TEX_WIDTH, i / LIGHTS_TEX_WIDTH),
                      0);
}

/*
 * Sample a point on an emissive voxel face visible from x, choosing the
 * voxel proportionally to its power.  Return false if nothing was sampled.
 */
bool sample_light(vec3 x, out vec3 wi, out float dist, out vec3 le,
                  out float pdf)
{
    int lo = 0, hi = u_lights_count - 1, mid, i, a, k = 0;
    int axes[3];
    vec3 c, d, p, n;
    float u, cos_l;

    if (u_lights_count == 0) return false;
    // Binary search in the CDF.
    u = rand();
    for (i = 0; i < 24 && lo < hi; i++) {
        mid = (lo + hi) / 2;
        if (get_light(u_lights_pos, mid).a > u) hi = mid;
        else lo = mid + 1;
    }
    c = get_light(u_lights_pos, lo).xyz;
    le = get_light(u_lights_emit, lo).rgb;

    // Pick a point on one of the faces toward x.
    d = x - (c + 0.5);
    for (a = 0; a < 3; a++) {
        if (abs(d[a]) > 0.5) axes[k++] = a;
    }
    if (k == 0) return false;
    a = axes[min(int(rand() * float(k)), k - 1)];
    p = c + vec3(rand(), rand(), rand());
    p[a] = c[a] + ((d[a] > 0.0) ? 1.0 : 0.0);
    n = vec3(0.0);
    n[a] = sign(d[a]);

    wi = p - x;
    dist = length(wi);
    wi /= dist;
    cos_l = dot(-wi, n);
    if (cos_l <= 0.0) return false;
    pdf = luminance(le) / u_lights_power / float(k) * dist * dist / cos_l;
    return pdf > 0.0;
}

/*
 * Sample a direction in the environment image, proportionally to its
 * luminance: first pick a row, then a pixel in that row.
 */
bool sample_env(out vec3 wi, out vec3 le, out float pdf)
{
    int lo, hi, mid, i, x, y;
    float u, theta, phi, lum;
    vec3 c;

    if (u_world_type != 3 || u_env_integral <= 0.0) return false;

    u = rand();
    lo = 0;
    hi = u_env_size.y - 1;
    for (i = 0; i < 16 && lo < hi; i++) {
        mid = (lo + hi) / 2;
        if (texelFetch(u_env_marg, ivec2(mid, 0), 0).r > u) hi = mid;
        else lo = mid + 1;
    }
    y = lo;

    u = rand();
    lo = 0;
    hi = u_env_size.x - 1;
    for (i = 0; i < 16 && lo < hi; i++) {
        mid = (lo + hi) / 2;
        if (texelFetch(u_env_cond, ivec2(mid, y), 0).r > u) hi = mid;
        else lo = mid + 1;
    }
    x = lo;

    c = texelFetch(u_env, ivec2(x, y), 0).rgb;
    lum = luminance(c);
    if (lum <= 0.0) return false;

    // Jitter inside the pixel.
    theta = (float(y) + rand()) / float(u_env_size.y) * PI;
    phi = ((float(x) + rand()) / float(u_env_size.x) - 0.5) * 2.0 * PI;
    wi = vec3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
    le = c * u_world_energy;
    pdf = lum * float(u_env_size.x) * float(u_env_size.y) /
          (u_env_integral * 2.0 * PI * PI);
    return pdf > 0.0;
}

// Pdf of sample_env for a given direction.
float env_pdf(vec3 d)
{
    float theta, phi, lum;
    int x, y;

    if (u_world_type != 3 || u_env_integral <= 0.0) return 0.0;
    theta = acos(clamp(d.z, -1.0, 1.0));
    phi = atan(d.y, d.x);
    x = clamp(int((phi / (2.0 * PI) + 0.5) * float(u_env_size.x)),
              0, u_env_size.x - 1);
    y = clamp(int(theta / PI * float(u_env_size.y)), 0, u_env_size.y - 1);
    lum = luminance(texelFetch(u_env, ivec2(x, y), 0).rgb);
    return lum * float(u_env_size.x) * float(u_env_size.y) /
           (u_env_integral * 2.0 * PI * PI);
}

// Pdf of sample_light from the vertex x, for an emissive voxel hit.
float light_pdf(vec3 x, hit_t hit, vec3 le)
{
    vec3 d, wi;
    float k, cos_l, dist;

    if (u_lights_count == 0 || hit.material == FLOOR_MATERIAL) return 0.0;
    wi = hit.pos - x;
    dist = length(wi);
    if (dist <= 0.0) return 0.0;
    wi /= dist;
    d = x - (floor(hit.pos - hit.normal * 0.5) + 0.5);
    k = float(abs(d.x) > 0.5) + float(abs(d.y) > 0.5) + float(abs(d.z) > 0.5);
    cos_l = abs(dot(wi, hit.normal));
    if (k == 0.0 || cos_l <= 0.0) return 0.0;
    return luminance(le) / u_lights_power / k * dist * dist / cos_l;
}

void main()
{
    ivec2 size = textureSize(u_prev, 0);
    ivec2 px = ivec2(gl_FragCoord.xy);
    float aspect = float(size.x) / float(size.y);
    float alpha = 1.0, q, path_roughness = 0.0, last_pdf = 0.0;
    float light_dist, light_p, ior, r, phi;
    vec2 ndc;
    vec3 ro, rd, wo, wi, w, l, pos, le, cz, last_vertex;
    vec3 radiance = vec3(0.0), throughput = vec3(1.0);
    hit_t hit;
    material_t m;
    int i, bounce = 0, medium = -1;
    bool specular;

    g_rng_state = (uint(px.x) * 73856093u) ^ (uint(px.y) * 19349663u) ^
                  (uint(u_seed) * 83492791u);
    rand();

    // Camera ray, with some jitter for the anti aliasing.
    ndc = (vec2(px) + vec2(rand(), rand())) / vec2(size) * 2.0 - 1.0;
    cz = -normalize(u_camera[2].xyz);
    if (u_ortho_size > 0.0) {
        rd = cz;
        ro = u_camera[3].xyz - rd * 1024.0 +
             normalize(u_camera[0].xyz) * ndc.x * u_ortho_size +
             normalize(u_camera[1].xyz) * ndc.y * u_ortho_size / aspect;
    } else {
        ro = u_camera[3].xyz;
        rd = normalize(cz +
            normalize(u_camera[0].xyz) * ndc.x * u_tan_fovy * aspect +
            normalize(u_camera[1].xyz) * ndc.y * u_tan_fovy);
        // Depth of field, with a thin lens model.
        if (u_aperture > 0.0) {
            pos = ro + rd * (u_focus / dot(rd, cz));
            r = 0.5 * u_aperture * sqrt(rand());
            phi = 2.0 * PI * rand();
            ro += normalize(u_camera[0].xyz) * r * cos(phi) +
                  normalize(u_camera[1].xyz) * r * sin(phi);
            rd = normalize(pos - ro);
        }
    }
    last_vertex = ro;

    for (i = 0; i < u_bounces + 16 && bounce <= u_bounces; i++) {
        if (!trace(ro, rd, medium, hit)) {
            if (i == 0 && u_world_type == 0) alpha = 0.0;
            if (medium == -1) {
                q = (bounce == 0) ? 1.0 : mis(last_pdf, env_pdf(rd));
                radiance += clamp_indirect(throughput * get_sky(rd) * q,
                                           bounce);
            }
            break;
        }

        // Refractive materials interfaces.
        ior = (hit.material >= 0 && hit.material != FLOOR_MATERIAL) ?
              get_ior(hit.material) : 0.0;
        if (medium >= 0 && hit.material >= 0 && ior == 0.0) {
            medium = -1; // Opaque surface inside a refractive material.
        } else if (medium >= 0 || ior > 0.0) {
            m = get_material((medium >= 0) ? medium : hit.material,
                             hit.color);
            if (sample_dielectric(rd, hit.normal,
                    ((medium >= 0) ? get_ior(medium) : 1.0) /
                    ((hit.material >= 0) ? ior : 1.0),
                    m.roughness, wi)) {
                // Tint the light when entering the material.
                if (medium == -1) throughput *= m.albedo;
                medium = hit.material;
                ro = hit.pos - hit.normal * 1e-3;
            } else {
                ro = hit.pos + hit.normal * 1e-3;
            }
            rd = wi;
            continue;
        }

        m = get_material(hit.material, hit.color);
        // Path regularization: the surfaces seen after a rough bounce are
        // made at least as rough, this removes most of the caustics noise.
        m.roughness = max(m.roughness, path_roughness);
        wo = -rd;
        pos = hit.pos + hit.normal * 1e-3;

        // Emission, weighted against the lights sampling of the last bounce.
        if (max3(m.emission) > 0.0) {
            q = (bounce == 0) ? 1.0 :
                mis(last_pdf, light_pdf(last_vertex, hit, m.emission));
            radiance += clamp_indirect(throughput * m.emission * q, bounce);
        }

        // Direct light from the sun.
        l = sample_cone(u_sun_dir, u_sun_cos);
        if (u_sun_intensity > 0.0 && dot(l, hit.normal) > 0.0) {
            w = transmittance(pos, l, 1e30);
            if (max3(w) > 0.0) {
                radiance += clamp_indirect(
                        throughput * eval_brdf(m, hit.normal, wo, l) *
                        u_sun_intensity * w, bounce);
            }
        }

        // Direct light from the emissive voxels.  At the last bounce there
        // is no BRDF sampling, so no need for the MIS weight.
        if (sample_light(pos, l, light_dist, le, light_p) &&
            dot(l, hit.normal) > 0.0) {
            w = transmittance(pos, l, light_dist - 2e-3);
            if (max3(w) > 0.0) {
                q = (bounce == u_bounces) ? 1.0 :
                    mis(light_p, brdf_pdf(m, hit.normal, wo, l));
                radiance += clamp_indirect(
                        throughput * eval_brdf(m, hit.normal, wo, l) * le *
                        w * q / light_p, bounce);
            }
        }

        // Direct light from the environment image.
        if (sample_env(l, le, light_p) && dot(l, hit.normal) > 0.0) {
            w = transmittance(pos, l, 1e30);
            if (max3(w) > 0.0) {
                q = (bounce == u_bounces) ? 1.0 :
                    mis(light_p, brdf_pdf(m, hit.normal, wo, l));
                radiance += clamp_indirect(
                        throughput * eval_brdf(m, hit.normal, wo, l) * le *
                        w * q / light_p, bounce);
            }
        }

        if (bounce == u_bounces) break;
        w = sample_brdf(m, hit.normal, wo, wi, specular, last_pdf);
        if (max3(w) <= 0.0) break;
        throughput *= w;
        path_roughness = max(path_roughness, specular ? m.roughness : 1.0);

        // Russian roulette.
        if (bounce >= 2) {
            q = clamp(max3(throughput), 0.05, 0.95);
            if (rand() > q) break;
            throughput /= q;
        }
        last_vertex = pos;
        ro = pos;
        rd = wi;
        bounce++;
    }

    // Clamp the fireflies.
    q = luminance(radiance);
    if (q > 32.0) radiance *= 32.0 / q;

    out_color = texelFetch(u_prev, px, 0) + vec4(radiance, alpha);
}

#endif // ACCUMULATE

#endif // FRAGMENT_SHADER
