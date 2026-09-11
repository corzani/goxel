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

#include <limits.h>

#if !defined(GLES2) && !defined(__APPLE__)

#define TABLE_FLAG (1u << 30)
#define FLOOR_MATERIAL 255 // Materials texture index used for the floor.
#define MAX_SAMPLES_PER_FRAME 16

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

static GLuint create_texture(GLenum target)
{
    GLuint tex;
    GL(glGenTextures(1, &tex));
    GL(glBindTexture(target, tex));
    GL(glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
    GL(glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
    GL(glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
    GL(glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE));
    return tex;
}

static void init_graphics(pathtracer_gpu_t *gpu)
{
    const float QUAD[] = {-1, -1, +1, -1, -1, +1, +1, +1};

    GL(glGenBuffers(1, &gpu->quad_buffer));
    GL(glBindBuffer(GL_ARRAY_BUFFER, gpu->quad_buffer));
    GL(glBufferData(GL_ARRAY_BUFFER, sizeof(QUAD), QUAD, GL_STATIC_DRAW));
    gpu->table_tex = create_texture(GL_TEXTURE_3D);
    gpu->atlas_tex = create_texture(GL_TEXTURE_3D);
    gpu->materials_tex = create_texture(GL_TEXTURE_2D);
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

static uint32_t get_volume_key(const scene_t *scene)
{
    const layer_t *layer;
    uint64_t k;
    uint32_t key = 0;
    int material;

    for (layer = scene->layers; layer; layer = layer->next) {
        if (!layer->visible || !layer->volume) continue;
        k = volume_get_key(layer->volume);
        material = get_layer_material(scene, layer);
        key = XXH32(&k, sizeof(k), key);
        key = XXH32(&material, sizeof(material), key);
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
    return key;
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
    bool empty;

    for (i = 0; i < 3; i++) {
        bmin[i] = INT_MAX;
        bmax[i] = INT_MIN;
    }

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
        }
        free(tile);
    }

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

static void update_buffers(pathtracer_gpu_t *gpu, int w, int h)
{
    int i;

    if (gpu->accum_tex[0] && gpu->w == w && gpu->h == h) return;
    for (i = 0; i < 2; i++) {
        if (!gpu->accum_tex[i])
            gpu->accum_tex[i] = create_texture(GL_TEXTURE_2D);
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
    gl_update_uniform(shader, "u_prev", 0);
    gl_update_uniform(shader, "u_table", 1);
    gl_update_uniform(shader, "u_atlas", 2);
    gl_update_uniform(shader, "u_materials", 3);
    gl_update_uniform(shader, "u_grid_origin", gpu->grid_origin);
    gl_update_uniform(shader, "u_grid_size", gpu->grid_size);
    gl_update_uniform(shader, "u_bounces", pt->bounces);

    gl_update_uniform(shader, "u_camera", camera->mat);
    gl_update_uniform(shader, "u_tan_fovy", tan(camera->fovy * DD2R / 2));
    gl_update_uniform(shader, "u_ortho_size",
                      camera->ortho ? camera->dist : 0.0);

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

// Tone map the accumulated samples into the pathtracer texture.
static void display(pathtracer_gpu_t *gpu, pathtracer_t *pt)
{
    const shader_define_t defines[] = {{"DISPLAY", true}, {}};
    gl_shader_t *shader;
    uint32_t key;

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
    gl_update_uniform(shader, "u_accum", 0);
    gl_update_uniform(shader, "u_samples", (float)gpu->samples);
    gl_update_uniform(shader, "u_exposure", pt->exposure);
    draw_quad(gpu);

    // Copy the final image into the buffer, so that it can be saved.
    if (pt->status != PT_FINISHED) return;
    key = XXH32(&gpu->scene_key, sizeof(gpu->scene_key), 0);
    key = XXH32(&pt->exposure, sizeof(pt->exposure), key);
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
    uint32_t volume_key, scene_key;
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
        GL(glBindFramebuffer(GL_FRAMEBUFFER, gpu->accum_fbo[gpu->accum_cur]));
        GL(glClearColor(0, 0, 0, 0));
        GL(glClear(GL_COLOR_BUFFER_BIT));
        gpu->samples = 0;
        gpu->display_key = 0;
        gpu->need_reset = false;
        pt->force_restart = false;
        if (pt->status == PT_FINISHED) pt->status = PT_RUNNING;
    }

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
    GL(glDeleteTextures(2, gpu->accum_tex));
    GL(glDeleteFramebuffers(2, gpu->accum_fbo));
    GL(glDeleteFramebuffers(1, &gpu->display_fbo));
    GL(glDeleteBuffers(1, &gpu->quad_buffer));
    free(gpu);
    pt->gpu = NULL;
}

#else // Dummy implementation.

bool pathtracer_gpu_is_supported(void) { return false; }
void pathtracer_gpu_iter(pathtracer_t *pt) {}
void pathtracer_gpu_reset(pathtracer_t *pt) {}
void pathtracer_gpu_release(pathtracer_t *pt) {}

#endif
