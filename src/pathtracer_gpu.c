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
 * GPU progressive path tracer.
 *
 * Alternative to the CPU yocto renderer that runs on the GPU and traces the
 * rays directly into the voxels instead of a triangle mesh, so that the
 * image converges in real time.  See data/shaders/pathtracer.glsl.
 *
 * The voxels of all the visible layers are merged into two 3D textures:
 *
 *   - An atlas with the voxels of all the non empty tiles: the sRGB color,
 *     and the material index + 1 in the alpha channel (0 for no voxel).
 *   - A table with one texel per tile, containing the position of the tile
 *     in the atlas (10 bits per axis) plus a flag bit, or zero if the tile
 *     is empty.
 *
 * Each frame we add a few samples into a float accumulation buffer (ping
 * pong between two textures), and then tone map the result into the
 * pathtracer texture that is shown in the view.
 */

#include "goxel.h"
#include "shader_cache.h"
#include "xxhash.h"
#include "stb_image.h"

#include <limits.h>

// Procedural skies: colors of the zenith, horizon and ground (linear).
static const struct {
    const char *name;
    float zenith[3];
    float horizon[3];
    float ground[3];
} SKIES[PT_SKY_COUNT] = {
    [PT_SKY_DAY] = {"Day",
        {0.20, 0.40, 0.85}, {0.80, 0.86, 0.92}, {0.35, 0.33, 0.30}},
    [PT_SKY_SUNSET] = {"Sunset",
        {0.08, 0.14, 0.38}, {1.00, 0.45, 0.15}, {0.25, 0.16, 0.10}},
    [PT_SKY_NIGHT] = {"Night",
        {0.010, 0.020, 0.060}, {0.04, 0.06, 0.12}, {0.010, 0.012, 0.020}},
    [PT_SKY_OVERCAST] = {"Overcast",
        {0.50, 0.54, 0.60}, {0.72, 0.74, 0.78}, {0.30, 0.30, 0.30}},
};

const char *pathtracer_sky_name(int sky)
{
    if (sky < 0 || sky >= PT_SKY_COUNT) return "";
    return SKIES[sky].name;
}

void pathtracer_sky_colors(int sky, float zenith[3], float horizon[3],
                           float ground[3])
{
    sky = clamp(sky, 0, PT_SKY_COUNT - 1);
    vec3_copy(SKIES[sky].zenith, zenith);
    vec3_copy(SKIES[sky].horizon, horizon);
    vec3_copy(SKIES[sky].ground, ground);
}

#if !defined(GLES2) && !defined(__APPLE__)

#define TABLE_FLAG (1u << 30)
#define FLOOR_MATERIAL 255 // Materials texture index used for the floor.
#define MAX_SAMPLES_PER_FRAME 16
#define LIGHTS_TEX_WIDTH 1024 // Must match the shader.
#define MAX_LIGHTS (1 << 20)
#define BLOOM_LEVELS 6
#define BLOOM_THRESHOLD 1.0

static const char *ATTR_NAMES[] = {"a_pos", NULL};

struct pathtracer_gpu {
    uint32_t    volume_key;
    uint32_t    scene_key;
    uint32_t    display_key;
    bool        has_grid;
    bool        need_reset;

    // Voxels data.
    GLuint      table_tex;
    GLuint      atlas_tex;
    GLuint      materials_tex;
    int         grid_origin[3]; // In voxels.
    int         grid_size[3];   // In tiles.

    // Visible emissive voxels, sampled for the direct lighting.
    int         (*lights)[4];   // Position and material index.
    int         nb_lights;
    int         lights_capacity;
    bool        has_lights;
    uint32_t    lights_key;
    int         lights_count;   // Number of lights in the textures.
    float       lights_power;   // Sum of the lights luminance.
    GLuint      lights_pos_tex;
    GLuint      lights_emit_tex;

    // Environment image, and its sampling tables.
    char        env_path[1024];
    GLuint      env_tex;
    GLuint      env_cond_tex;   // Cumulated luminance of each row.
    GLuint      env_marg_tex;   // Cumulated luminance of the rows.
    int         env_size[2];
    float       env_integral;   // Sum of the weighted luminances.

    // Bloom mipmaps: [0] downsampled, [1] upsampled.
    GLuint      bloom_tex[2][BLOOM_LEVELS];
    GLuint      bloom_fbo[2][BLOOM_LEVELS];
    int         bloom_size[BLOOM_LEVELS][2];

    // Accumulation buffers.
    int         w, h;
    GLuint      accum_tex[2];
    GLuint      accum_fbo[2];
    int         accum_cur;  // Index of the buffer with the current samples.
    int         samples;
    int         samples_per_frame;
    int         seed;

    GLuint      display_fbo;
    GLuint      quad_buffer;
};

// The visible layers and their materials.
typedef struct {
    const layer_t       *layers;
    const material_t    *materials[FLOOR_MATERIAL];
    int                 nb_materials;
} scene_t;

// Used to collect the tiles of all the layers.
typedef struct {
    UT_hash_handle  hh;
    int             pos[3];
    int             slot;
} tile_item_t;

bool pathtracer_gpu_is_supported(void)
{
    static int supported = -1;
    const char *str;
    int major = 0, minor = 0;

    if (supported == -1) {
        str = (const char*)glGetString(GL_SHADING_LANGUAGE_VERSION);
        if (str) sscanf(str, "%d.%d", &major, &minor);
        supported = (major * 100 + minor >= 330);
        if (!supported)
            LOG_I("GPU path tracer not supported (GLSL %s)", str ?: "?");
    }
    return supported;
}

static GLuint create_texture(GLenum target, GLenum filter)
{
    GLuint tex;
    GL(glGenTextures(1, &tex));
    GL(glBindTexture(target, tex));
    GL(glTexParameteri(target, GL_TEXTURE_MIN_FILTER, filter));
    GL(glTexParameteri(target, GL_TEXTURE_MAG_FILTER, filter));
    GL(glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
    return tex;
}

static void init_graphics(pathtracer_gpu_t *gpu)
{
    const float QUAD[] = {-1, -1, +1, -1, -1, +1, +1, +1};
    const float zero[4] = {};

    GL(glGenBuffers(1, &gpu->quad_buffer));
    GL(glBindBuffer(GL_ARRAY_BUFFER, gpu->quad_buffer));
    GL(glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD, GL_STATIC_DRAW));
    gpu->table_tex = create_texture(GL_TEXTURE_3D, GL_NEAREST);
    gpu->atlas_tex = create_texture(GL_TEXTURE_3D, GL_NEAREST);
    gpu->materials_tex = create_texture(GL_TEXTURE_2D, GL_NEAREST);
    gpu->lights_pos_tex = create_texture(GL_TEXTURE_2D, GL_NEAREST);
    gpu->lights_emit_tex = create_texture(GL_TEXTURE_2D, GL_NEAREST);

    gpu->env_tex = create_texture(GL_TEXTURE_2D, GL_LINEAR);
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, 1, 1, 0, GL_RGB, GL_FLOAT,
                    zero));
    gpu->env_cond_tex = create_texture(GL_TEXTURE_2D, GL_NEAREST);
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT,
                    zero));
    gpu->env_marg_tex = create_texture(GL_TEXTURE_2D, GL_NEAREST);
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, 1, 1, 0, GL_RED, GL_FLOAT,
                    zero));
    GL(glGenFramebuffers(2, gpu->accum_fbo));
    GL(glGenFramebuffers(1, &gpu->display_fbo));
    gpu->samples_per_frame = 1;
}

static void collect_scene(scene_t *scene)
{
    const layer_t *layer;
    int i;

    scene->layers = goxel_get_render_layers(false);
    for (layer = scene->layers; layer; layer = layer->next) {
        if (!layer->visible || !layer->volume) continue;
        for (i = 0; i < scene->nb_materials; i++) {
            if (scene->materials[i] == layer->material) break;
        }
        if (i == scene->nb_materials && i < FLOOR_MATERIAL)
            scene->materials[scene->nb_materials++] = layer->material;
    }
}

static int get_layer_material(const scene_t *scene, const layer_t *layer)
{
    int i;
    for (i = 0; i < scene->nb_materials; i++) {
        if (scene->materials[i] == layer->material) return i;
    }
    return FLOOR_MATERIAL - 1; // Too many materials.
}

static bool is_emissive(const material_t *mat)
{
    return mat && (mat->emission[0] > 0 || mat->emission[1] > 0 ||
                   mat->emission[2] > 0);
}

// Key of the voxels grid.  Also depends on the emissive materials, since
// the lights list is built with the grid.
static uint32_t get_volume_key(const scene_t *scene)
{
    const layer_t *layer;
    uint64_t k;
    uint32_t key = 0;
    int material;
    bool emissive;

    for (layer = scene->layers; layer; layer = layer->next) {
        if (!layer->visible || !layer->volume) continue;
        k = volume_get_key(layer->volume);
        material = get_layer_material(scene, layer);
        emissive = is_emissive(layer->material);
        key = XXH32(&k, sizeof(k), key);
        key = XXH32(&material, sizeof(material), key);
        key = XXH32(&emissive, sizeof(emissive), key);
    }
    return key;
}

static uint32_t get_lights_key(const scene_t *scene, uint32_t key)
{
    const float zero[3] = {};
    int i;

    for (i = 0; i < scene->nb_materials; i++) {
        key = XXH32(scene->materials[i] ? scene->materials[i]->emission : zero,
                    sizeof(zero), key);
    }
    return key;
}

static uint32_t get_material_key(const material_t *mat, uint32_t key)
{
    const material_t default_mat = MATERIAL_DEFAULT;
    uint32_t k;

    mat = mat ?: &default_mat;
    k = material_get_hash(mat);
    key = XXH32(&k, sizeof(k), key);
    return XXH32(mat->emission, sizeof(mat->emission), key);
}

// Key of everything that invalidates the accumulated samples.
static uint32_t get_scene_key(const pathtracer_t *pt, const scene_t *scene,
                              uint32_t key)
{
    const camera_t *camera = goxel.image->active_camera;
    float light_dir[3];
    int i;

    for (i = 0; i < scene->nb_materials; i++)
        key = get_material_key(scene->materials[i], key);
    key = get_material_key(pt->floor.material, key);
    key = XXH32(&pt->floor.type, sizeof(pt->floor.type), key);
    key = XXH32(pt->floor.color, sizeof(pt->floor.color), key);
    key = XXH32(pt->floor.size, sizeof(pt->floor.size), key);
    key = XXH32(&pt->world, sizeof(pt->world), key);
    key = XXH32(&pt->bounces, sizeof(pt->bounces), key);
    key = XXH32(&pt->sun_angle, sizeof(pt->sun_angle), key);
    key = XXH32(&pt->w, sizeof(pt->w), key);
    key = XXH32(&pt->h, sizeof(pt->h), key);
    key = XXH32(goxel.image->box, sizeof(goxel.image->box), key);

    render_get_light_dir(&goxel.rend, light_dir);
    key = XXH32(light_dir, sizeof(light_dir), key);
    key = XXH32(&goxel.rend.light.intensity,
                sizeof(goxel.rend.light.intensity), key);

    key = XXH32(camera->mat, sizeof(camera->mat), key);
    key = XXH32(&camera->ortho, sizeof(camera->ortho), key);
    key = XXH32(&camera->dist, sizeof(camera->dist), key);
    key = XXH32(&camera->fovy, sizeof(camera->fovy), key);
    key = XXH32(&camera->aperture, sizeof(camera->aperture), key);
    key = XXH32(&camera->focus, sizeof(camera->focus), key);
    return key;
}

// Add the visible emissive voxels of a tile to the lights list.
static void add_tile_lights(pathtracer_gpu_t *gpu, const int pos[3],
                            uint8_t (*voxels)[4], const bool *emissive)
{
    const int N = TILE_SIZE;
    int i, x, y, z, material;

    for (i = 0; i < N * N * N; i++) {
        material = voxels[i][3];
        if (!material || !emissive[material - 1]) continue;
        x = i % N;
        y = i / N % N;
        z = i / (N * N);
        // Skip the voxels surrounded by other voxels of the tile.
        if (    x > 0 && x < N - 1 && y > 0 && y < N - 1 &&
                z > 0 && z < N - 1 &&
                voxels[i - 1][3] && voxels[i + 1][3] &&
                voxels[i - N][3] && voxels[i + N][3] &&
                voxels[i - N * N][3] && voxels[i + N * N][3])
            continue;
        if (gpu->nb_lights >= MAX_LIGHTS) return;
        if (gpu->nb_lights >= gpu->lights_capacity) {
            gpu->lights_capacity = max(1024, gpu->lights_capacity * 2);
            gpu->lights = realloc(gpu->lights,
                    gpu->lights_capacity * sizeof(*gpu->lights));
        }
        gpu->lights[gpu->nb_lights][0] = pos[0] + x;
        gpu->lights[gpu->nb_lights][1] = pos[1] + y;
        gpu->lights[gpu->nb_lights][2] = pos[2] + z;
        gpu->lights[gpu->nb_lights][3] = material - 1;
        gpu->nb_lights++;
    }
}

// Merge the voxels of all the layers and upload them into the textures.
static void update_grid(pathtracer_gpu_t *gpu, const scene_t *scene)
{
    const int N = TILE_SIZE;
    tile_item_t *tiles = NULL, *tile, *tmp;
    volume_iterator_t iter;
    const layer_t *layer;
    const uint8_t *data;
    uint8_t voxels[TILE_SIZE * TILE_SIZE * TILE_SIZE][4];
    uint32_t *table;
    int i, n = 0, pos[3], bmin[3], bmax[3], atlas[3], block[3], t[3];
    int max_size, max_blocks, material;
    bool empty, emissive[FLOOR_MATERIAL];

    for (i = 0; i < 3; i++) {
        bmin[i] = INT_MAX;
        bmax[i] = INT_MIN;
    }
    for (i = 0; i < FLOOR_MATERIAL; i++)
        emissive[i] = is_emissive(scene->materials[i]);
    gpu->nb_lights = 0;

    // Collect all the tiles positions.
    for (layer = scene->layers; layer; layer = layer->next) {
        if (!layer->visible || !layer->volume) continue;
        iter = volume_get_iterator(layer->volume, VOLUME_ITER_TILES);
        while (volume_iter(&iter, pos)) {
            HASH_FIND(hh, tiles, pos, sizeof(pos), tile);
            if (tile) continue;
            tile = calloc(1, sizeof(*tile));
            memcpy(tile->pos, pos, sizeof(pos));
            tile->slot = n++;
            HASH_ADD(hh, tiles, pos, sizeof(tile->pos), tile);
            for (i = 0; i < 3; i++) {
                bmin[i] = min(bmin[i], pos[i]);
                bmax[i] = max(bmax[i], pos[i]);
            }
        }
    }

    GL(glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE, &max_size));
    max_blocks = min(max_size / N, 1024);
    atlas[0] = atlas[1] = clamp((int)ceil(cbrt(n)), 1, max_blocks);
    atlas[2] = max(1, (n + atlas[0] * atlas[1] - 1) / (atlas[0] * atlas[1]));
    memset(gpu->grid_origin, 0, sizeof(gpu->grid_origin));
    memset(gpu->grid_size, 0, sizeof(gpu->grid_size));
    for (i = 0; n && i < 3; i++) {
        gpu->grid_origin[i] = bmin[i];
        gpu->grid_size[i] = (bmax[i] - bmin[i]) / N + 1;
    }
    if (    atlas[2] > max_blocks ||
            gpu->grid_size[0] > max_size ||
            gpu->grid_size[1] > max_size ||
            gpu->grid_size[2] > max_size) {
        LOG_W("Volume too large for the GPU path tracer");
        memset(gpu->grid_size, 0, sizeof(gpu->grid_size));
        atlas[0] = atlas[1] = atlas[2] = 1;
    }

    GL(glBindTexture(GL_TEXTURE_3D, gpu->atlas_tex));
    GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_SRGB8_ALPHA8,
                    atlas[0] * N, atlas[1] * N, atlas[2] * N, 0,
                    GL_RGBA, GL_UNSIGNED_BYTE, NULL));
    table = calloc(max(1, gpu->grid_size[0] * gpu->grid_size[1] *
                          gpu->grid_size[2]), sizeof(*table));

    HASH_ITER(hh, tiles, tile, tmp) {
        HASH_DEL(tiles, tile);
        if (gpu->grid_size[0] == 0) {
            free(tile);
            continue;
        }
        // Merge the layers, the last ones on top.
        memset(voxels, 0, sizeof(voxels));
        empty = true;
        for (layer = scene->layers; layer; layer = layer->next) {
            if (!layer->visible || !layer->volume) continue;
            data = volume_get_tile_data(layer->volume, NULL, tile->pos, NULL);
            if (!data) continue;
            material = get_layer_material(scene, layer);
            for (i = 0; i < N * N * N; i++) {
                if (data[i * 4 + 3] < 127) continue;
                memcpy(voxels[i], &data[i * 4], 3);
                voxels[i][3] = material + 1;
                empty = false;
            }
        }
        if (!empty) {
            block[0] = tile->slot % atlas[0];
            block[1] = tile->slot / atlas[0] % atlas[1];
            block[2] = tile->slot / (atlas[0] * atlas[1]);
            GL(glTexSubImage3D(GL_TEXTURE_3D, 0,
                               block[0] * N, block[1] * N, block[2] * N,
                               N, N, N, GL_RGBA, GL_UNSIGNED_BYTE, voxels));
            for (i = 0; i < 3; i++)
                t[i] = (tile->pos[i] - gpu->grid_origin[i]) / N;
            table[t[0] + t[1] * gpu->grid_size[0] +
                  t[2] * gpu->grid_size[0] * gpu->grid_size[1]] =
                TABLE_FLAG | block[0] | block[1] << 10 | block[2] << 20;
            add_tile_lights(gpu, tile->pos, voxels, emissive);
        }
        free(tile);
    }
    if (gpu->nb_lights >= MAX_LIGHTS)
        LOG_W("Too many emissive voxels for the GPU path tracer");

    GL(glBindTexture(GL_TEXTURE_3D, gpu->table_tex));
    GL(glTexImage3D(GL_TEXTURE_3D, 0, GL_R32UI,
                    max(gpu->grid_size[0], 1),
                    max(gpu->grid_size[1], 1),
                    max(gpu->grid_size[2], 1), 0,
                    GL_RED_INTEGER, GL_UNSIGNED_INT, table));
    free(table);
}

static void update_materials(pathtracer_gpu_t *gpu, const pathtracer_t *pt,
                             const scene_t *scene)
{
    const material_t default_mat = MATERIAL_DEFAULT;
    const material_t *mat;
    float data[3 * 256][4] = {}; // Rows: base color, emission, metal/rough.
    float floor_color[3];
    int i;

    for (i = 0; i < 256; i++) {
        if (i == FLOOR_MATERIAL)
            mat = pt->floor.material;
        else if (i < scene->nb_materials)
            mat = scene->materials[i];
        else
            continue;
        mat = mat ?: &default_mat;
        vec4_copy(mat->base_color, data[i]);
        vec3_copy(mat->emission, data[256 + i]);
        data[512 + i][0] = mat->metallic;
        data[512 + i][1] = mat->roughness;
        data[512 + i][2] = mat->ior;
    }

    // The floor color is picked in sRGB, like the voxels colors.
    srgb8_to_rgb(pt->floor.color, floor_color);
    for (i = 0; i < 3; i++)
        data[FLOOR_MATERIAL][i] *= floor_color[i];
    data[FLOOR_MATERIAL][3] = 1;

    GL(glBindTexture(GL_TEXTURE_2D, gpu->materials_tex));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, 256, 3, 0,
                    GL_RGBA, GL_FLOAT, data));
}

// Upload the lights positions, CDF (by luminance) and emission.
static void update_lights(pathtracer_gpu_t *gpu, const scene_t *scene)
{
    const material_t default_mat = MATERIAL_DEFAULT;
    const material_t *mat;
    float (*pos)[4], (*emit)[4], total = 0;
    int i, n = gpu->nb_lights, w, h;

    w = clamp(n, 1, LIGHTS_TEX_WIDTH);
    h = max(1, (n + LIGHTS_TEX_WIDTH - 1) / LIGHTS_TEX_WIDTH);
    pos = calloc(w * h, sizeof(*pos));
    emit = calloc(w * h, sizeof(*emit));
    for (i = 0; i < n; i++) {
        mat = scene->materials[gpu->lights[i][3]] ?: &default_mat;
        vec3_copy(mat->emission, emit[i]);
        emit[i][3] = 1;
        total += 0.2126 * mat->emission[0] + 0.7152 * mat->emission[1] +
                 0.0722 * mat->emission[2];
        pos[i][0] = gpu->lights[i][0];
        pos[i][1] = gpu->lights[i][1];
        pos[i][2] = gpu->lights[i][2];
        pos[i][3] = total;
    }
    for (i = 0; total > 0 && i < n; i++)
        pos[i][3] /= total;
    if (total > 0) pos[n - 1][3] = 1;
    gpu->lights_count = (total > 0) ? n : 0;
    gpu->lights_power = total;

    GL(glBindTexture(GL_TEXTURE_2D, gpu->lights_pos_tex));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                    GL_RGBA, GL_FLOAT, pos));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->lights_emit_tex));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                    GL_RGBA, GL_FLOAT, emit));
    free(pos);
    free(emit);
}

/*
 * Load the environment image and compute the tables used to sample it:
 * for each row the cumulated luminance of its pixels, and the cumulated
 * luminance of the rows.  The luminances are weighted by the solid angle
 * of the pixels, that shrinks toward the poles.
 */
static void update_env(pathtracer_gpu_t *gpu, const pathtracer_t *pt)
{
    float *data = NULL, *cond = NULL, *marg = NULL;
    float sin_theta, lum, total = 0, prev, row;
    int w = 0, h = 0, n, x, y;

    snprintf(gpu->env_path, sizeof(gpu->env_path), "%s", pt->world.image);
    gpu->env_integral = 0;
    memset(gpu->env_size, 0, sizeof(gpu->env_size));

    if (*pt->world.image)
        data = stbi_loadf(pt->world.image, &w, &h, &n, 3);
    if (!data) {
        if (*pt->world.image)
            LOG_W("Cannot open environment image %s", pt->world.image);
        return;
    }

    cond = calloc(w * h, sizeof(*cond));
    marg = calloc(h, sizeof(*marg));
    for (y = 0; y < h; y++) {
        sin_theta = sin(M_PI * (y + 0.5) / h);
        for (x = 0; x < w; x++) {
            lum = 0.2126 * data[(y * w + x) * 3 + 0] +
                  0.7152 * data[(y * w + x) * 3 + 1] +
                  0.0722 * data[(y * w + x) * 3 + 2];
            total += max(lum, 0.f) * sin_theta;
            cond[y * w + x] = total;
        }
        marg[y] = total;
    }
    // Normalize each row, and the rows themselves.  'prev' keeps the value
    // of the previous row before it gets normalized.
    prev = 0;
    for (y = 0; y < h; y++) {
        row = marg[y] - prev;
        for (x = 0; x < w; x++) {
            cond[y * w + x] = (row > 0) ? (cond[y * w + x] - prev) / row : 1;
        }
        prev = marg[y];
        marg[y] = (total > 0) ? marg[y] / total : 1;
    }

    gpu->env_size[0] = w;
    gpu->env_size[1] = h;
    gpu->env_integral = total;

    GL(glBindTexture(GL_TEXTURE_2D, gpu->env_tex));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32F, w, h, 0, GL_RGB, GL_FLOAT,
                    data));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->env_cond_tex));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT,
                    cond));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->env_marg_tex));
    GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, h, 1, 0, GL_RED, GL_FLOAT,
                    marg));
    LOG_I("Environment image %s (%dx%d)", pt->world.image, w, h);

    stbi_image_free(data);
    free(cond);
    free(marg);
}

static void update_buffers(pathtracer_gpu_t *gpu, int w, int h)
{
    int i, j;

    if (gpu->accum_tex[0] && gpu->w == w && gpu->h == h) return;

    for (i = 0; i < BLOOM_LEVELS; i++) {
        gpu->bloom_size[i][0] = max(1, (w / 2) >> i);
        gpu->bloom_size[i][1] = max(1, (h / 2) >> i);
        for (j = 0; j < 2; j++) {
            if (!gpu->bloom_tex[j][i]) {
                gpu->bloom_tex[j][i] = create_texture(GL_TEXTURE_2D,
                                                      GL_LINEAR);
                GL(glGenFramebuffers(1, &gpu->bloom_fbo[j][i]));
            }
            GL(glBindTexture(GL_TEXTURE_2D, gpu->bloom_tex[j][i]));
            GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F,
                            gpu->bloom_size[i][0], gpu->bloom_size[i][1], 0,
                            GL_RGBA, GL_FLOAT, NULL));
            GL(glBindFramebuffer(GL_FRAMEBUFFER, gpu->bloom_fbo[j][i]));
            GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                      GL_TEXTURE_2D, gpu->bloom_tex[j][i], 0));
            assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
                   GL_FRAMEBUFFER_COMPLETE);
        }
    }

    for (i = 0; i < 2; i++) {
        if (!gpu->accum_tex[i])
            gpu->accum_tex[i] = create_texture(GL_TEXTURE_2D, GL_LINEAR);
        GL(glBindTexture(GL_TEXTURE_2D, gpu->accum_tex[i]));
        GL(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, w, h, 0,
                        GL_RGBA, GL_FLOAT, NULL));
        GL(glBindFramebuffer(GL_FRAMEBUFFER, gpu->accum_fbo[i]));
        GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                  GL_TEXTURE_2D, gpu->accum_tex[i], 0));
        assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
               GL_FRAMEBUFFER_COMPLETE);
    }
    gpu->w = w;
    gpu->h = h;
    gpu->need_reset = true;
}

static void draw_quad(pathtracer_gpu_t *gpu)
{
    int i;

    GL(glBindBuffer(GL_ARRAY_BUFFER, gpu->quad_buffer));
    // Other renderers can leave some attributes enabled.
    for (i = 1; i < 16; i++)
        GL(glDisableVertexAttribArray(i));
    GL(glEnableVertexAttribArray(0));
    GL(glVertexAttribPointer(0, 2, GL_FLOAT, false, 0, NULL));
    GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, 4));
    GL(glDisableVertexAttribArray(0));
}

static void accumulate(pathtracer_gpu_t *gpu, const pathtracer_t *pt, int n)
{
    const shader_define_t defines[] = {{"ACCUMULATE", true}, {}};
    const camera_t *camera = goxel.image->active_camera;
    const image_t *image = goxel.image;
    gl_shader_t *shader;
    float light_dir[3], world_color[3], floor_rect[4], center[3] = {};
    float sky_zenith[3], sky_horizon[3], sky_ground[3];
    float floor_z = 0;
    int i;

    shader = shader_get("pathtracer", defines, ATTR_NAMES, NULL);
    GL(glUseProgram(shader->prog));

    GL(glActiveTexture(GL_TEXTURE1));
    GL(glBindTexture(GL_TEXTURE_3D, gpu->table_tex));
    GL(glActiveTexture(GL_TEXTURE2));
    GL(glBindTexture(GL_TEXTURE_3D, gpu->atlas_tex));
    GL(glActiveTexture(GL_TEXTURE3));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->materials_tex));
    GL(glActiveTexture(GL_TEXTURE4));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->lights_pos_tex));
    GL(glActiveTexture(GL_TEXTURE5));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->lights_emit_tex));
    GL(glActiveTexture(GL_TEXTURE6));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->env_tex));
    GL(glActiveTexture(GL_TEXTURE7));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->env_cond_tex));
    GL(glActiveTexture(GL_TEXTURE8));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->env_marg_tex));
    gl_update_uniform(shader, "u_prev", 0);
    gl_update_uniform(shader, "u_table", 1);
    gl_update_uniform(shader, "u_atlas", 2);
    gl_update_uniform(shader, "u_materials", 3);
    gl_update_uniform(shader, "u_lights_pos", 4);
    gl_update_uniform(shader, "u_lights_emit", 5);
    gl_update_uniform(shader, "u_lights_count", gpu->lights_count);
    gl_update_uniform(shader, "u_lights_power", gpu->lights_power);
    gl_update_uniform(shader, "u_env", 6);
    gl_update_uniform(shader, "u_env_cond", 7);
    gl_update_uniform(shader, "u_env_marg", 8);
    gl_update_uniform(shader, "u_env_size", gpu->env_size);
    gl_update_uniform(shader, "u_env_integral", gpu->env_integral);
    gl_update_uniform(shader, "u_grid_origin", gpu->grid_origin);
    gl_update_uniform(shader, "u_grid_size", gpu->grid_size);
    gl_update_uniform(shader, "u_bounces", pt->bounces);

    gl_update_uniform(shader, "u_camera", camera->mat);
    gl_update_uniform(shader, "u_tan_fovy", tan(camera->fovy * DD2R / 2));
    gl_update_uniform(shader, "u_ortho_size",
                      camera->ortho ? camera->dist : 0.0);
    gl_update_uniform(shader, "u_aperture",
                      camera->ortho ? 0.0 : camera->aperture);
    gl_update_uniform(shader, "u_focus",
                      (camera->focus > 0) ? camera->focus : camera->dist);

    render_get_light_dir(&goxel.rend, light_dir);
    gl_update_uniform(shader, "u_sun_dir", light_dir);
    gl_update_uniform(shader, "u_sun_intensity", goxel.rend.light.intensity);
    gl_update_uniform(shader, "u_sun_cos", cos(pt->sun_angle * DD2R / 2));

    // Like the CPU renderer, the world color is used as a linear value.
    for (i = 0; i < 3; i++)
        world_color[i] = pt->world.color[i] / 255.f * pt->world.energy;
    gl_update_uniform(shader, "u_world_type", pt->world.type);
    gl_update_uniform(shader, "u_world_energy", pt->world.energy);
    gl_update_uniform(shader, "u_world_color", world_color);

    pathtracer_sky_colors(pt->world.sky, sky_zenith, sky_horizon, sky_ground);
    gl_update_uniform(shader, "u_sky_zenith", sky_zenith);
    gl_update_uniform(shader, "u_sky_horizon", sky_horizon);
    gl_update_uniform(shader, "u_sky_ground", sky_ground);

    // Floor at the bottom of the image box, same as the CPU renderer.
    if (!box_is_null(image->box)) {
        vec3_copy(image->box[3], center);
        floor_z = image->box[3][2] - image->box[2][2];
    }
    floor_rect[0] = center[0] - pt->floor.size[0] / 2.f;
    floor_rect[1] = center[1] - pt->floor.size[1] / 2.f;
    floor_rect[2] = center[0] + pt->floor.size[0] / 2.f;
    floor_rect[3] = center[1] + pt->floor.size[1] / 2.f;
    gl_update_uniform(shader, "u_floor", pt->floor.type == PT_FLOOR_PLANE);
    gl_update_uniform(shader, "u_floor_z", floor_z);
    gl_update_uniform(shader, "u_floor_rect", floor_rect);

    GL(glViewport(0, 0, gpu->w, gpu->h));
    for (i = 0; i < n; i++) {
        GL(glBindFramebuffer(GL_FRAMEBUFFER,
                             gpu->accum_fbo[1 - gpu->accum_cur]));
        GL(glActiveTexture(GL_TEXTURE0));
        GL(glBindTexture(GL_TEXTURE_2D, gpu->accum_tex[gpu->accum_cur]));
        gl_update_uniform(shader, "u_seed", gpu->seed++);
        draw_quad(gpu);
        gpu->accum_cur = 1 - gpu->accum_cur;
        gpu->samples++;
    }
}

/*
 * Compute the bloom from the accumulated samples, with a chain of
 * downsampled images, then upsampled back while adding each level.
 * Return the final bloom texture (at half the resolution).
 */
static GLuint render_bloom(pathtracer_gpu_t *gpu)
{
    const shader_define_t down_defines[] = {{"BLOOM_DOWN", true}, {}};
    const shader_define_t up_defines[] = {{"BLOOM_UP", true}, {}};
    gl_shader_t *shader;
    float size[2];
    int i;

    shader = shader_get("pathtracer", down_defines, ATTR_NAMES, NULL);
    GL(glUseProgram(shader->prog));
    GL(glActiveTexture(GL_TEXTURE0));
    gl_update_uniform(shader, "u_src", 0);
    for (i = 0; i < BLOOM_LEVELS; i++) {
        vec2_set(size, gpu->bloom_size[i][0], gpu->bloom_size[i][1]);
        GL(glBindFramebuffer(GL_FRAMEBUFFER, gpu->bloom_fbo[0][i]));
        GL(glViewport(0, 0, size[0], size[1]));
        GL(glBindTexture(GL_TEXTURE_2D, (i == 0) ?
                         gpu->accum_tex[gpu->accum_cur] :
                         gpu->bloom_tex[0][i - 1]));
        gl_update_uniform(shader, "u_dst_size", size);
        gl_update_uniform(shader, "u_scale",
                          (i == 0) ? 1.0 / max(gpu->samples, 1) : 1.0);
        gl_update_uniform(shader, "u_threshold",
                          (i == 0) ? BLOOM_THRESHOLD : 0.0);
        draw_quad(gpu);
    }

    shader = shader_get("pathtracer", up_defines, ATTR_NAMES, NULL);
    GL(glUseProgram(shader->prog));
    gl_update_uniform(shader, "u_src", 0);
    gl_update_uniform(shader, "u_base", 1);
    for (i = BLOOM_LEVELS - 2; i >= 0; i--) {
        vec2_set(size, gpu->bloom_size[i][0], gpu->bloom_size[i][1]);
        GL(glBindFramebuffer(GL_FRAMEBUFFER, gpu->bloom_fbo[1][i]));
        GL(glViewport(0, 0, size[0], size[1]));
        GL(glActiveTexture(GL_TEXTURE0));
        GL(glBindTexture(GL_TEXTURE_2D, (i == BLOOM_LEVELS - 2) ?
                         gpu->bloom_tex[0][i + 1] :
                         gpu->bloom_tex[1][i + 1]));
        GL(glActiveTexture(GL_TEXTURE1));
        GL(glBindTexture(GL_TEXTURE_2D, gpu->bloom_tex[0][i]));
        gl_update_uniform(shader, "u_dst_size", size);
        draw_quad(gpu);
    }
    return gpu->bloom_tex[1][0];
}

// Tone map the accumulated samples into the pathtracer texture.
static void display(pathtracer_gpu_t *gpu, pathtracer_t *pt)
{
    const shader_define_t defines[] = {{"DISPLAY", true}, {}};
    gl_shader_t *shader;
    uint32_t key;
    GLuint bloom_tex = gpu->bloom_tex[1][0];

    if (pt->bloom > 0) bloom_tex = render_bloom(gpu);

    shader = shader_get("pathtracer", defines, ATTR_NAMES, NULL);
    GL(glBindFramebuffer(GL_FRAMEBUFFER, gpu->display_fbo));
    // Always attach, since the texture can have been recreated.
    GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, pt->texture->tex, 0));
    assert(glCheckFramebufferStatus(GL_FRAMEBUFFER) ==
           GL_FRAMEBUFFER_COMPLETE);
    GL(glViewport(0, 0, gpu->w, gpu->h));
    GL(glUseProgram(shader->prog));
    GL(glActiveTexture(GL_TEXTURE0));
    GL(glBindTexture(GL_TEXTURE_2D, gpu->accum_tex[gpu->accum_cur]));
    GL(glActiveTexture(GL_TEXTURE1));
    GL(glBindTexture(GL_TEXTURE_2D, bloom_tex));
    gl_update_uniform(shader, "u_accum", 0);
    gl_update_uniform(shader, "u_samples", (float)gpu->samples);
    gl_update_uniform(shader, "u_exposure", pt->exposure);
    gl_update_uniform(shader, "u_bloom", 1);
    gl_update_uniform(shader, "u_bloom_intensity", max(pt->bloom, 0));
    draw_quad(gpu);

    // Copy the final image into the buffer, so that it can be saved.
    if (pt->status != PT_FINISHED) return;
    key = XXH32(&gpu->scene_key, sizeof(gpu->scene_key), 0);
    key = XXH32(&pt->exposure, sizeof(pt->exposure), key);
    key = XXH32(&pt->bloom, sizeof(pt->bloom), key);
    key = XXH32(&gpu->samples, sizeof(gpu->samples), key);
    if (key == gpu->display_key) return;
    gpu->display_key = key;
    GL(glReadPixels(0, 0, gpu->w, gpu->h, GL_RGBA, GL_UNSIGNED_BYTE,
                    pt->buf));
}

void pathtracer_gpu_iter(pathtracer_t *pt)
{
    pathtracer_gpu_t *gpu;
    scene_t scene = {};
    uint32_t volume_key, scene_key, lights_key;
    int n;

    if (!pt->gpu) pt->gpu = calloc(1, sizeof(*pt->gpu));
    gpu = pt->gpu;
    if (!gpu->quad_buffer) init_graphics(gpu);

    GL(glDisable(GL_SCISSOR_TEST));
    GL(glDisable(GL_DEPTH_TEST));
    GL(glDisable(GL_CULL_FACE));
    GL(glDisable(GL_BLEND));

    collect_scene(&scene);
    volume_key = get_volume_key(&scene);
    if (!gpu->has_grid || volume_key != gpu->volume_key) {
        update_grid(gpu, &scene);
        gpu->volume_key = volume_key;
        gpu->has_grid = true;
    }
    update_buffers(gpu, pt->w, pt->h);

    scene_key = get_scene_key(pt, &scene, volume_key);
    if (scene_key != gpu->scene_key || pt->force_restart || gpu->need_reset) {
        gpu->scene_key = scene_key;
        update_materials(gpu, pt, &scene);
        if (    pt->world.type == PT_WORLD_IMAGE &&
                strcmp(gpu->env_path, pt->world.image) != 0) {
            update_env(gpu, pt);
        }
        lights_key = get_lights_key(&scene, volume_key);
        if (!gpu->has_lights || lights_key != gpu->lights_key) {
            update_lights(gpu, &scene);
            gpu->lights_key = lights_key;
            gpu->has_lights = true;
        }
        GL(glBindFramebuffer(GL_FRAMEBUFFER, gpu->accum_fbo[gpu->accum_cur]));
        GL(glClearColor(0, 0, 0, 0));
        GL(glClear(GL_COLOR_BUFFER_BIT));
        gpu->samples = 0;
        gpu->display_key = 0;
        gpu->need_reset = false;
        pt->force_restart = false;
        if (pt->status == PT_FINISHED) pt->status = PT_RUNNING;
    }

    // The number of samples can be increased after the rendering finished.
    if (pt->status == PT_FINISHED && gpu->samples < pt->num_samples)
        pt->status = PT_RUNNING;

    if (gpu->samples < pt->num_samples) {
        // Adapt the number of samples per frame to keep the UI responsive.
        if (goxel.delta_time < 1.0 / 45)
            gpu->samples_per_frame = min(gpu->samples_per_frame + 1,
                                         MAX_SAMPLES_PER_FRAME);
        if (goxel.delta_time > 1.0 / 20)
            gpu->samples_per_frame = max(gpu->samples_per_frame / 2, 1);
        n = min(gpu->samples_per_frame, pt->num_samples - gpu->samples);
        accumulate(gpu, pt, n);
    }
    pt->samples = gpu->samples;
    if (pt->samples >= pt->num_samples && pt->status == PT_RUNNING)
        pt->status = PT_FINISHED;

    display(gpu, pt);
    GL(glActiveTexture(GL_TEXTURE0));
}

void pathtracer_gpu_reset(pathtracer_t *pt)
{
    if (pt->gpu) pt->gpu->need_reset = true;
}

void pathtracer_gpu_release(pathtracer_t *pt)
{
    pathtracer_gpu_t *gpu = pt->gpu;
    if (!gpu) return;
    GL(glDeleteTextures(1, &gpu->table_tex));
    GL(glDeleteTextures(1, &gpu->atlas_tex));
    GL(glDeleteTextures(1, &gpu->materials_tex));
    GL(glDeleteTextures(1, &gpu->lights_pos_tex));
    GL(glDeleteTextures(1, &gpu->lights_emit_tex));
    GL(glDeleteTextures(1, &gpu->env_tex));
    GL(glDeleteTextures(1, &gpu->env_cond_tex));
    GL(glDeleteTextures(1, &gpu->env_marg_tex));
    GL(glDeleteTextures(2, gpu->accum_tex));
    GL(glDeleteFramebuffers(2, gpu->accum_fbo));
    GL(glDeleteTextures(BLOOM_LEVELS * 2, (GLuint*)gpu->bloom_tex));
    GL(glDeleteFramebuffers(BLOOM_LEVELS * 2, (GLuint*)gpu->bloom_fbo));
    GL(glDeleteFramebuffers(1, &gpu->display_fbo));
    GL(glDeleteBuffers(1, &gpu->quad_buffer));
    free(gpu->lights);
    free(gpu);
    pt->gpu = NULL;
}

#else // Dummy implementation.

bool pathtracer_gpu_is_supported(void) { return false; }
void pathtracer_gpu_iter(pathtracer_t *pt) {}
void pathtracer_gpu_reset(pathtracer_t *pt) {}
void pathtracer_gpu_release(pathtracer_t *pt) {}

#endif
