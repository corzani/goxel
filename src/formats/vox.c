/* Goxel 3D voxels editor
 *
 * copyright (c) 2016 Guillaume Chereau <guillaume@noctua-software.com>
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

// Magica voxel vox format support.

#include "file_format.h"
#include "goxel.h"

#include <limits.h>
#include <errno.h>

static const uint32_t VOX_DEFAULT_PALETTE[256];

// Conversion from MagicaVoxel lens aperture to a lens diameter in voxels.
#define VOX_APERTURE_SCALE 10

static inline void hexcolor(uint32_t v, uint8_t out[4])
{
    out[0] = (v >> 24) & 0xff;
    out[1] = (v >> 16) & 0xff;
    out[2] = (v >>  8) & 0xff;
    out[3] = (v >>  0) & 0xff;
}

#define READ(type, file) \
    ({ type v; size_t r = fread(&v, sizeof(v), 1, file); \
       if (r != 1) goto error; \
       v;})

#define WRITE(type, v, file) \
    ({ type v_ = v; fwrite(&v_, sizeof(v_), 1, file);})

#define FILE_ERROR(msg) do { LOG_E("%s", msg); goto error; } while (0);

// Import the old magica voxel file format:
//
// d, h, w, <data>, <palette>
static int vox_import_old(const char *path)
{
    FILE *file;
    int w, h, d, i;
    uint8_t *voxels = NULL;
    uint8_t (*palette)[4] = NULL;
    uint8_t (*cube)[4] = NULL;
    int ret = -1;
    size_t size;

    file = fopen(path, "rb");
    if (!file) return -1;
    d = READ(uint32_t, file);
    h = READ(uint32_t, file);
    w = READ(uint32_t, file);

    size = w * h * d;
    if (size * sizeof(*cube) >= SIZE_MAX / 4) goto error;
    voxels = calloc(size, 1);
    palette = calloc(256, sizeof(*palette));
    cube = calloc(size, sizeof(*cube));
    if (!cube) goto error;
    for (i = 0; i < w * h * d; i++) {
        voxels[i] = READ(uint8_t, file);
    }
    for (i = 0; i < 256; i++) {
        palette[i][0] = READ(uint8_t, file);
        palette[i][1] = READ(uint8_t, file);
        palette[i][2] = READ(uint8_t, file);
        palette[i][3] = 255;
    }
    memset(palette[255], 0, 4);

    for (i = 0; i < w * h * d; i++) {
        if (voxels[i] == 255) continue;
        memcpy(cube[i], palette[voxels[i]], 4);
    }

    volume_blit(goxel.image->active_layer->volume, (uint8_t*)cube,
              -w / 2, -h / 2, -d / 2, w, h, d, NULL);
    ret = 0;

error:
    free(palette);
    free(voxels);
    free(cube);
    fclose(file);
    return ret;
}

// MagicaVoxel material types.
enum {
    VOX_MATL_DIFFUSE = 0,
    VOX_MATL_METAL,
    VOX_MATL_GLASS,
    VOX_MATL_EMIT,
    VOX_MATL_MEDIA,
};

typedef struct node node_t;
struct node {
    node_t *children;
    node_t *next, *prev;

    char id[4];
    node_t *parent; // Logical parent.

    int node_id;
    int nb_children;
    int *children_ids;

    union {
        struct {
            int w, h, d;
        } size;
        struct {
            uint8_t (*values)[4];
        } rgba;
        struct {
            int nb;
            uint8_t *values;
        } xyzi;
        struct {
            int id;
            int child_id;
            int nb_frames;
            int tr[3];
            bool has_rot;
            int rot[3][3];
            bool has_trans;
            int trans[3];
            char name[256];
            int layer_id;
            bool hidden;
        } ntrn;
        struct {
            bool hidden;
        } ngrp;
        struct {
            int id;
            bool hidden;
        } layr;
        struct {
            int id;
            int nb_models;
            int model_id;
        } nshp;
        struct {
            int id;
            int type;
            float emit;
            float flux;
            float metal;
            float rough;
            float trans;
            float density;
            float ri;
        } matl;
        struct {
            int id;
            int nb;
            char keys[16][32];
            char values[16][64];
        } dict; // rOBJ and rCAM attributes.
    };
};

// Voxels of all the shapes using a given non diffuse material.  We create
// one layer for each, since goxel materials are per layer.
typedef struct {
    material_t  material;
    char        name[64];
    volume_t    *volumes[2]; // Voxels of the visible and hidden shapes.
} vox_group_t;

static int read_string(FILE *file, char **out)
{
    int size, r;
    size = READ(int32_t, file);
    *out = calloc(size + 1, 1);
    r = fread(*out, 1, size, file);
    if (r != size) {
        return -1;
    }
    return size;

error:
    return -1;
}

static void read_dict(FILE *file, void *user,
                      void (*callback)(void *user, const char *key, int size,
                                       const char *value))
{
    int nb, i, size;
    char *key, *value;
    nb = READ(int32_t, file);
    for (i = 0; i < nb; i++) {
        size = read_string(file, &key);
        if (size < 0) goto error;
        size = read_string(file, &value);
        if (size < 0) goto error;
        if (callback) callback(user, key, size, value);
        free(key);
        free(value);
    }

error:
    return;
}

static void on_trn_dict(void *user, const char *key, int size,
                        const char *value)
{
    node_t *node = user;
    if (strcmp(key, "_name") == 0) {
        snprintf(node->ntrn.name, sizeof(node->ntrn.name), "%.*s", size, value);
    }
    if (strcmp(key, "_hidden") == 0)
        node->ntrn.hidden = atoi(value);
}

static void on_grp_dict(void *user, const char *key, int size,
                        const char *value)
{
    node_t *node = user;
    if (strcmp(key, "_hidden") == 0)
        node->ngrp.hidden = atoi(value);
}

static void on_layr_dict(void *user, const char *key, int size,
                         const char *value)
{
    node_t *node = user;
    if (strcmp(key, "_hidden") == 0)
        node->layr.hidden = atoi(value);
}

static void on_trn_frame_dict(void *user, const char *key, int size,
                              const char *value)
{
    node_t *node = user;
    int v, x, y, z;
    if (strcmp(key, "_r") == 0) {
        node->ntrn.has_rot = true;
        v = atoi(value);
        x = (v >> 0) & 3;
        y = (v >> 2) & 3;
        z = 3 - x - y;
        node->ntrn.rot[x][0] = ((v >> 4) & 1) ? -1 : +1;
        node->ntrn.rot[y][1] = ((v >> 5) & 1) ? -1 : +1;
        node->ntrn.rot[z][2] = ((v >> 6) & 1) ? -1 : +1;
    }
    if (strcmp(key, "_t") == 0) {
        node->ntrn.has_trans = true;
        sscanf(value, "%d %d %d", &x, &y, &z);
        node->ntrn.trans[0] = x;
        node->ntrn.trans[1] = y;
        node->ntrn.trans[2] = z;
    }
}

static void on_matl_dict(void *user, const char *key, int size,
                         const char *value)
{
    node_t *node = user;
    if (strcmp(key, "_type") == 0) {
        if (strcmp(value, "_metal") == 0) node->matl.type = VOX_MATL_METAL;
        if (strcmp(value, "_glass") == 0) node->matl.type = VOX_MATL_GLASS;
        if (strcmp(value, "_emit") == 0) node->matl.type = VOX_MATL_EMIT;
        if (strcmp(value, "_media") == 0 || strcmp(value, "_cloud") == 0)
            node->matl.type = VOX_MATL_MEDIA;
    }
    if (strcmp(key, "_emit") == 0) node->matl.emit = atof(value);
    if (strcmp(key, "_flux") == 0) node->matl.flux = atof(value);
    if (strcmp(key, "_metal") == 0) node->matl.metal = atof(value);
    if (strcmp(key, "_rough") == 0) node->matl.rough = atof(value);
    if (strcmp(key, "_trans") == 0) node->matl.trans = atof(value);
    if (strcmp(key, "_d") == 0) node->matl.density = atof(value);
    if (strcmp(key, "_ri") == 0) node->matl.ri = atof(value);
    // Newer files use '_ior', that stores the index of refraction minus one.
    if (strcmp(key, "_ior") == 0) node->matl.ri = 1 + atof(value);
}

static void on_store_dict(void *user, const char *key, int size,
                          const char *value)
{
    node_t *node = user;
    int n = node->dict.nb;
    if (n >= ARRAY_SIZE(node->dict.keys)) return;
    snprintf(node->dict.keys[n], sizeof(node->dict.keys[n]), "%s", key);
    snprintf(node->dict.values[n], sizeof(node->dict.values[n]), "%s", value);
    node->dict.nb++;
}

static bool node_is(const node_t *node, const char *id)
{
    return strncmp(node->id, id, 4) == 0;
}

static void free_node(node_t *node)
{
    node_t *child, *tmp;

    if (node_is(node, "RGBA"))
        free(node->rgba.values);
    if (node_is(node, "XYZI"))
        free(node->xyzi.values);

    DL_FOREACH_SAFE(node->children, child, tmp) {
        DL_DELETE(node->children, child);
        free_node(child);
    }

    free(node->children_ids);
    free(node);
}

static node_t *read_node(FILE *file)
{
    int i, r, size, children_size;
    node_t *node, *child, *child2;
    long fpos;

    node = calloc(1, sizeof(*node));
    node->node_id = -1;
    r = fread(node->id, 1, 4, file);
    if (r != 4) goto error;

    size = READ(uint32_t, file);
    children_size = READ(uint32_t, file);

    fpos = ftell(file);

    if (strncmp(node->id, "MAIN", 4) == 0) {
        // Nothing to do.
    }
    else if (strncmp(node->id, "SIZE", 4) == 0) {
        node->size.w = READ(uint32_t, file);
        node->size.h = READ(uint32_t, file);
        node->size.d = READ(uint32_t, file);
    }
    else if (strncmp(node->id, "RGBA", 4) == 0) {
        node->rgba.values = malloc(4 * 256);
        for (i = 1; i < 256; i++) {
            node->rgba.values[i][0] = READ(uint8_t, file);
            node->rgba.values[i][1] = READ(uint8_t, file);
            node->rgba.values[i][2] = READ(uint8_t, file);
            node->rgba.values[i][3] = READ(uint8_t, file);
        }
        // Skip the last value!
        for (i = 0; i < 4; i++) READ(uint8_t, file);
    }
    else if (strncmp(node->id, "XYZI", 4) == 0) {
        node->xyzi.nb = READ(uint32_t, file);
        node->xyzi.values = calloc(node->xyzi.nb, 4);
        for (i = 0; i < node->xyzi.nb; i++) {
            node->xyzi.values[i * 4 + 0] = READ(uint8_t, file);
            node->xyzi.values[i * 4 + 1] = READ(uint8_t, file);
            node->xyzi.values[i * 4 + 2] = READ(uint8_t, file);
            node->xyzi.values[i * 4 + 3] = READ(uint8_t, file);
        }
    }
    else if (strncmp(node->id, "nTRN", 4) == 0) {
        node->node_id = READ(int32_t, file);
        read_dict(file, node, on_trn_dict);
        node->nb_children = 1;
        node->children_ids = calloc(1, sizeof(int));
        *node->children_ids = READ(int32_t, file);
        READ(int32_t, file); // Reserved.
        node->ntrn.layer_id = READ(int32_t, file);
        node->ntrn.nb_frames = READ(int32_t, file);
        for (i = 0; i < node->ntrn.nb_frames; i++) {
            read_dict(file, node, on_trn_frame_dict);
        }
    }
    else if (strncmp(node->id, "nSHP", 4) == 0) {
        node->node_id = READ(int32_t, file);
        read_dict(file, NULL, NULL);
        node->nshp.nb_models = READ(int32_t, file);
        for (i = 0; i < node->nshp.nb_models; i++) {
            node->nshp.model_id = READ(int32_t, file);
            read_dict(file, NULL, NULL);
        }
    }
    else if (strncmp(node->id, "MATL", 4) == 0) {
        node->matl.id = READ(int32_t, file);
        read_dict(file, node, on_matl_dict);
    }
    else if (strncmp(node->id, "rOBJ", 4) == 0) {
        read_dict(file, node, on_store_dict);
    }
    else if (strncmp(node->id, "rCAM", 4) == 0) {
        node->dict.id = READ(int32_t, file);
        read_dict(file, node, on_store_dict);
    }
    else if (strncmp(node->id, "LAYR", 4) == 0) {
        node->layr.id = READ(int32_t, file);
        read_dict(file, node, on_layr_dict);
    }
    else if (strncmp(node->id, "nGRP", 4) == 0) {
        node->node_id = READ(int32_t, file);
        read_dict(file, node, on_grp_dict);
        node->nb_children = READ(int32_t, file);
        node->children_ids = calloc(node->nb_children, sizeof(int));
        for (i = 0; i < node->nb_children; i++) {
            node->children_ids[i] = READ(int32_t, file);
        }
    }

    if (ftell(file) < fpos + size) {
        fseek(file, fpos + size, SEEK_SET);
    }

    while (ftell(file) < fpos + size + children_size) {
        child = read_node(file);
        if (!child) continue;
        DL_APPEND(node->children, child);
    }


    // Set the parents.
    DL_FOREACH(node->children, child) {
        for (i = 0; i < child->nb_children; i++) {
            DL_FOREACH(node->children, child2) {
                if (child2->node_id == child->children_ids[i]) {
                    child2->parent = child;
                }
            }
        }
    }

    return node;
error:
    free_node(node);
    return NULL;
}

// Test if a node is hidden, or in a hidden group or layer.
static bool node_is_hidden(const node_t *node, const node_t *tree)
{
    const node_t *layr;

    for (; node; node = node->parent) {
        if (node_is(node, "nGRP") && node->ngrp.hidden) return true;
        if (!node_is(node, "nTRN")) continue;
        if (node->ntrn.hidden) return true;
        DL_FOREACH(tree->children, layr) {
            if (    node_is(layr, "LAYR") &&
                    layr->layr.id == node->ntrn.layer_id &&
                    layr->layr.hidden)
                return true;
        }
    }
    return false;
}

static void node_apply_mat(const node_t *node, float mat[4][4])
{
    float rot[4][4] = MAT4_IDENTITY;
    int i, j;

    if (node->parent) {
        node_apply_mat(node->parent, mat);
    }

    if (node_is(node, "nTRN") && node->ntrn.has_trans) {
        mat4_itranslate(mat, node->ntrn.trans[0],
                             node->ntrn.trans[1],
                             node->ntrn.trans[2]);
    }
    if (node_is(node, "nTRN") && node->ntrn.has_rot) {
        for (i = 0; i < 3; i++) {
            for (j = 0; j < 3; j++) {
                rot[i][j] = node->ntrn.rot[i][j];
            }
        }
        mat4_imul(mat, rot);
    }
}

static const node_t *node_get_ntrn(const node_t *node)
{
    if (!node) {
        return NULL;
    }
    if (node_is(node, "nTRN")) {
        return node;
    }
    return node_get_ntrn(node->parent);
}

static void get_palette_color(const node_t *rgba, int c, uint8_t color[4])
{
    if (rgba)
        memcpy(color, rgba->rgba.values[c], 4);
    else
        hexcolor(VOX_DEFAULT_PALETTE[c], color);
}

/*
 * Convert a MagicaVoxel material into goxel material properties.
 * Return false for the diffuse materials, that just use the layer material.
 *
 * This is only an approximation, the rendering models are different.
 */
static bool matl_to_material(const node_t *matl, const uint8_t color[4],
                             material_t *mat, const char **type_name)
{
    float rgb[3];

    *mat = MATERIAL_DEFAULT;
    switch (matl->matl.type) {
    case VOX_MATL_EMIT:
        // _emit is the amount (0 to 1) and _flux the power (0 to 4, shown
        // as 1 to 5 in MagicaVoxel).  Goxel emission is an absolute color.
        // The power saturates above flux 2: a single emitter set to the
        // highest values would otherwise be bright enough to wash out the
        // whole scene.
        srgb8_to_rgb(color, rgb);
        vec3_mul(rgb, 4 * matl->matl.emit * pow(4, fmin(matl->matl.flux, 2)),
                 mat->emission);
        mat->metallic = 0;
        mat->roughness = matl->matl.rough;
        *type_name = "Emit";
        return true;
    case VOX_MATL_GLASS:
        mat->metallic = 0;
        mat->roughness = matl->matl.rough;
        mat->base_color[3] = 1 - matl->matl.trans;
        mat->ior = (matl->matl.ri > 0) ? matl->matl.ri : 1.5;
        *type_name = "Glass";
        return true;
    case VOX_MATL_METAL:
        mat->metallic = matl->matl.metal;
        mat->roughness = matl->matl.rough;
        *type_name = "Metal";
        return true;
    case VOX_MATL_MEDIA:
        // No volumetric rendering, use a semi transparent material.
        mat->metallic = 0;
        mat->roughness = 1;
        mat->base_color[3] = clamp(matl->matl.density * 5, 0.05, 1);
        *type_name = "Media";
        return true;
    default:
        return false;
    }
}

// Return the index of the group for a given material, creating it if needed.
static int get_group(vox_group_t *groups, int *nb, const material_t *mat,
                     const char *type_name)
{
    int i;
    vox_group_t *group;

    for (i = 0; i < *nb; i++) {
        group = &groups[i];
        if (    group->material.metallic == mat->metallic &&
                group->material.roughness == mat->roughness &&
                memcmp(group->material.base_color, mat->base_color,
                       sizeof(mat->base_color)) == 0 &&
                memcmp(group->material.emission, mat->emission,
                       sizeof(mat->emission)) == 0)
            return i;
    }
    group = &groups[(*nb)++];
    group->material = *mat;
    snprintf(group->name, sizeof(group->name), "%s %d", type_name, *nb);
    group->volumes[0] = volume_new();
    group->volumes[1] = volume_new();
    return *nb - 1;
}

/*
 * Import a model into a new layer.
 *
 * shape  - The scene shape instance of the model, used for the position,
 *          name and visibility.  NULL for the old files without scene.
 * first  - If set, use the current layer instead of creating a new one.
 */
static int import_layer(image_t *image, bool first,
                        const node_t *size, const node_t *xyzi,
                        const node_t *rgba, const node_t *shape,
                        const node_t *tree, const int palette_group[256],
                        vox_group_t *groups)
{
    int i, x, y, z, c, g, pos[3];
    layer_t *layer;
    uint8_t color[4];
    volume_iterator_t iter = {0};
    volume_t *group_volumes[256] = {};
    const node_t *ntrn;
    float mat[4][4] = MAT4_IDENTITY;
    bool hidden = shape && node_is_hidden(shape, tree);

    layer = first ? image->active_layer : image_add_layer(image, NULL);
    layer->visible = !hidden;

    for (i = 0; i < xyzi->xyzi.nb; i++) {
        x = xyzi->xyzi.values[i * 4 + 0];
        y = xyzi->xyzi.values[i * 4 + 1];
        z = xyzi->xyzi.values[i * 4 + 2];
        c = xyzi->xyzi.values[i * 4 + 3];
        pos[0] = x - size->size.w / 2;
        pos[1] = y - size->size.h / 2;
        pos[2] = z - size->size.d / 2;
        if (!c) continue; // Not sure what c == 0 means.
        get_palette_color(rgba, c, color);
        g = palette_group[c];
        if (g == -1) {
            volume_set_at(layer->volume, &iter, pos, color);
        } else {
            if (!group_volumes[g]) group_volumes[g] = volume_new();
            volume_set_at(group_volumes[g], NULL, pos, color);
        }
    }

    // Apply the transformation.
    // XXX: would be better to properly support layer transformations!
    if (shape) {
        node_apply_mat(shape, mat);
        volume_move(layer->volume, mat);
        ntrn = node_get_ntrn(shape);
        if (ntrn && *ntrn->ntrn.name) {
            snprintf(layer->name, sizeof(layer->name), "%s", ntrn->ntrn.name);
        }
    }

    // Move the voxels with special materials to their group.
    for (g = 0; g < 256; g++) {
        if (!group_volumes[g]) continue;
        if (shape) volume_move(group_volumes[g], mat);
        volume_merge(groups[g].volumes[hidden], group_volumes[g], MODE_OVER,
                     NULL);
        volume_delete(group_volumes[g]);
    }

    return 0;
}

static const char *dict_get(const node_t *node, const char *key)
{
    int i;
    if (!node) return NULL;
    for (i = 0; i < node->dict.nb; i++) {
        if (strcmp(node->dict.keys[i], key) == 0)
            return node->dict.values[i];
    }
    return NULL;
}

static float dict_get_float(const node_t *node, const char *key,
                            float default_value)
{
    const char *v = dict_get(node, key);
    return v ? atof(v) : default_value;
}

// Return the rOBJ node of a given type.
static const node_t *get_robj(const node_t *tree, const char *type)
{
    const node_t *node;
    const char *v;
    DL_FOREACH(tree->children, node) {
        if (!node_is(node, "rOBJ")) continue;
        v = dict_get(node, "_type");
        if (v && strcmp(v, type) == 0) return node;
    }
    return NULL;
}

/*
 * Import the MagicaVoxel camera and render settings.
 * This is only an approximation, the renderers are different.
 */
static void import_render_settings(image_t *image, const node_t *tree,
                                   const char *path)
{
    const node_t *rcam, *robj;
    const char *v, *name;
    char dir[512], env_path[1024];
    FILE *file;
    camera_t *camera, *first_camera = NULL;
    pathtracer_t *pt = &goxel.pathtracer;
    float focus[3], angle[3], radius, yaw, pitch;
    float x[3], y[3], z[3];
    int i, color[3];

    // Add all the saved cameras, the first one becomes the active camera.
    DL_FOREACH(tree->children, rcam) {
        if (!node_is(rcam, "rCAM")) continue;
        radius = dict_get_float(rcam, "_radius", 0);
        if (radius <= 0) continue;
        memset(focus, 0, sizeof(focus));
        memset(angle, 0, sizeof(angle));
        v = dict_get(rcam, "_focus");
        if (v) sscanf(v, "%f %f %f", &focus[0], &focus[1], &focus[2]);
        v = dict_get(rcam, "_angle");
        if (v) sscanf(v, "%f %f %f", &angle[0], &angle[1], &angle[2]);

        camera = image_add_camera(image, NULL);
        first_camera = first_camera ?: camera;
        snprintf(camera->name, sizeof(camera->name), "MagicaVoxel %d",
                 rcam->dict.id);
        v = dict_get(rcam, "_mode");
        camera->ortho = v && (strcmp(v, "orth") == 0 ||
                              strcmp(v, "iso") == 0);
        camera->fovy = dict_get_float(rcam, "_fov", 45);
        camera->dist = radius;
        camera->focus = radius;
        camera->aperture = VOX_APERTURE_SCALE *
            dict_get_float(get_robj(tree, "_lens"), "_aperture", 0);

        // Orbit camera around the focus point.
        yaw = angle[0] * DD2R;
        pitch = angle[1] * DD2R;
        vec3_set(z, sin(yaw) * cos(pitch), -cos(yaw) * cos(pitch),
                 sin(pitch));
        vec3_cross(VEC(0, 0, 1), z, x);
        vec3_normalize(x, x);
        vec3_cross(z, x, y);
        mat4_set_identity(camera->mat);
        vec3_copy(x, camera->mat[0]);
        vec3_copy(y, camera->mat[1]);
        vec3_copy(z, camera->mat[2]);
        vec3_addk(focus, z, radius, camera->mat[3]);
    }
    if (first_camera) image->active_camera = first_camera;

    // Sun.
    if ((robj = get_robj(tree, "_inf"))) {
        goxel.rend.light.intensity = dict_get_float(robj, "_i", 1);
        v = dict_get(robj, "_angle");
        if (v && sscanf(v, "%f %f", &angle[0], &angle[1]) == 2) {
            // Elevation and azimuth, in degrees.
            goxel.rend.light.pitch = (90 - angle[0]) * DD2R;
            goxel.rend.light.yaw = (angle[1] + 90) * DD2R;
            goxel.rend.light.fixed = false;
        }
        pt->sun_angle = dict_get_float(robj, "_area", 0.035) * DR2D;
    }

    // Environment: sky, or uniform (also used when it is an image, since
    // the image is not stored in the file).
    if ((robj = get_robj(tree, "_env"))) {
        if (dict_get_float(robj, "_mode", 0) == 0) {
            pt->world.type = PT_WORLD_SKY;
            pt->world.sky = PT_SKY_DAY;
            pt->world.energy = 1;
        } else {
            robj = get_robj(tree, "_uni");
            pt->world.type = PT_WORLD_UNIFORM;
            pt->world.energy = dict_get_float(robj, "_i", 1);
            v = dict_get(robj, "_k");
            if (v && sscanf(v, "%d %d %d",
                            &color[0], &color[1], &color[2]) == 3) {
                for (i = 0; i < 3; i++)
                    pt->world.color[i] = clamp(color[i], 0, 255);
            }
        }
    }

    /*
     * Environment image.  The image itself is not in the file, so we only
     * use it if we find it next to the model.
     */
    robj = get_robj(tree, "_ibl");
    v = robj ? dict_get(robj, "_path") : NULL;
    if (v && *v && path) {
        name = v + strlen(v);
        while (name > v && name[-1] != '/' && name[-1] != '\\') name--;
        path_dirname(path, dir, sizeof(dir));
        snprintf(env_path, sizeof(env_path), "%s/%s", dir, name);
        file = fopen(env_path, "rb");
        if (file) {
            fclose(file);
            pt->world.type = PT_WORLD_IMAGE;
            pt->world.energy = dict_get_float(robj, "_i", 1);
            snprintf(pt->world.image, sizeof(pt->world.image), "%s",
                     env_path);
            LOG_I("Using environment image %s", env_path);
        } else {
            LOG_I("Environment image %s not found next to the model", name);
        }
    }

    if ((robj = get_robj(tree, "_film")))
        pt->exposure = dict_get_float(robj, "_expo", 1);
    if ((robj = get_robj(tree, "_bloom")))
        pt->bloom = dict_get_float(robj, "_mix", 0);
    if ((robj = get_robj(tree, "_setting"))) {
        pt->floor.type = dict_get_float(robj, "_ground", 0) ?
                         PT_FLOOR_PLANE : PT_FLOOR_NONE;
    }
}

static int vox_import(const file_format_t *format, image_t *image,
                      const char *path)
{
    FILE *file;
    char magic[4];
    int r, i, j, c, version, nb_groups = 0, nb_models = 0;
    int palette_group[256];
    node_t *tree, *size_n, *rgba_n, *node;
    const node_t *matls[256] = {};
    const node_t **models;
    vox_group_t *groups;
    material_t material, *active_material, *mat;
    const char *type_name;
    uint8_t color[4];
    layer_t *layer;
    bool first = true;

    path = path ?: sys_open_file_dialog("Open", NULL, format->exts,
                                        format->exts_desc);
    if (!path) return -1;
    file = fopen(path, "rb");
    if (!file) return -1;
    r = fread(magic, 1, 4, file);
    if (r != 4) FILE_ERROR("Cannot read file");

    if (strncmp(magic, "VOX ", 4) != 0) {
        LOG_D("Old style magica voxel file");
        fclose(file);
        return vox_import_old(path);
    }

    if (strncmp(magic, "VOX ", 4) != 0) FILE_ERROR("Wrong magic string");
    version = READ(uint32_t, file);
    if (version != 150) LOG_W("Magica voxel file version %d!", version);
    tree = read_node(file);
    if (!tree) goto error;

    // Get the palette.
    DL_FOREACH(tree->children, rgba_n) {
        if (strncmp(rgba_n->id, "RGBA", 4) == 0) break;
    }

    // Get the materials, and the group of each palette color.
    DL_FOREACH(tree->children, node) {
        if (node_is(node, "MATL") && node->matl.id > 0 && node->matl.id < 256)
            matls[node->matl.id] = node;
    }
    groups = calloc(256, sizeof(*groups));
    for (c = 0; c < 256; c++) {
        palette_group[c] = -1;
        if (!matls[c]) continue;
        get_palette_color(rgba_n, c, color);
        if (matl_to_material(matls[c], color, &material, &type_name))
            palette_group[c] = get_group(groups, &nb_groups, &material,
                                         type_name);
    }

    // Get the models: the ('size', 'xyzi') chunks in the main chunk.
    models = calloc(256, sizeof(*models));
    DL_FOREACH(tree->children, size_n) {
        if (!node_is(size_n, "SIZE") || !size_n->next) continue;
        if (!node_is(size_n->next, "XYZI")) continue;
        if (nb_models % 256 == 0)
            models = realloc(models, (nb_models + 256) * sizeof(*models));
        models[nb_models++] = size_n;
    }

    // Create one layer for each shape instance of the scene.
    DL_FOREACH(tree->children, node) {
        if (!node_is(node, "nSHP")) continue;
        if (node->nshp.model_id < 0 || node->nshp.model_id >= nb_models)
            continue;
        size_n = (node_t*)models[node->nshp.model_id];
        import_layer(image, first, size_n, size_n->next, rgba_n, node, tree,
                     palette_group, groups);
        first = false;
    }
    // Old files without scene: one layer per model.
    for (i = 0; first && i < nb_models; i++) {
        import_layer(image, i == 0, models[i], models[i]->next, rgba_n, NULL,
                     tree, palette_group, groups);
    }
    free(models);

    // Create one layer and material for each non diffuse material used,
    // with a second hidden layer for the voxels of the hidden shapes.
    active_material = image->active_material;
    for (i = 0; i < nb_groups; i++) {
        mat = NULL;
        for (j = 0; j < 2; j++) {
            if (!volume_is_empty(groups[i].volumes[j])) {
                if (!mat) {
                    groups[i].material.ref = 1;
                    snprintf(groups[i].material.name,
                             sizeof(groups[i].material.name),
                             "%s", groups[i].name);
                    mat = image_add_material(
                            image, material_copy(&groups[i].material));
                }
                layer = image_add_layer(image, NULL);
                snprintf(layer->name, sizeof(layer->name), "%s",
                         groups[i].name);
                layer->material = mat;
                layer->visible = (j == 0);
                volume_set(layer->volume, groups[i].volumes[j]);
            }
            volume_delete(groups[i].volumes[j]);
        }
    }
    /*
     * The diffuse materials are not metallic, and have their own roughness.
     * Without an explicit material the layers would use the goxel default
     * one, that is slightly metallic, and would show specular reflections
     * that are not in the original scene.
     */
    for (c = 1; c < 256; c++) {
        if (matls[c] && matls[c]->matl.type == VOX_MATL_DIFFUSE) break;
    }
    if (c < 256) {
        material = MATERIAL_DEFAULT;
        material.metallic = 0;
        material.roughness = matls[c]->matl.rough;
        snprintf(material.name, sizeof(material.name), "Diffuse");
        mat = image_add_material(image, material_copy(&material));
        DL_FOREACH(image->layers, layer) {
            if (!layer->material) layer->material = mat;
        }
    }

    image->active_material = active_material;
    free(groups);

    import_render_settings(image, tree, path);
    free_node(tree);
    fclose(file);

    return 0;

error:
    fclose(file);
    return -1;
}

static int get_color_index(uint8_t v[4], uint8_t (*palette)[4], bool exact)
{
    const uint8_t *c;
    int i, dist, best = -1, best_dist = 1024;
    for (i = 1; i < 255; i++) {
        c = palette[i];
        dist = abs((int)c[0] - (int)v[0]) +
               abs((int)c[1] - (int)v[1]) +
               abs((int)c[2] - (int)v[2]);
        if (dist == 0) return i;
        if (exact) continue;
        if (dist < best_dist) {
            best_dist = dist;
            best = i;
        }
    }
    return best;
}

static int voxel_cmp(const void *a_, const void *b_)
{
    const uint8_t *a = a_;
    const uint8_t *b = b_;
    if (a[2] != b[2]) return cmp(a[2], b[2]);
    if (a[1] != b[1]) return cmp(a[1], b[1]);
    if (a[0] != b[0]) return cmp(a[0], b[0]);
    return 0;
}

static int vox_export(const file_format_t *format, const image_t *image,
                      const char *path)
{
    FILE *file;
    int children_size, nb_vox = 0, i, pos[3];
    int xmin = INT_MAX, ymin = INT_MAX, zmin = INT_MAX;
    int xmax = INT_MIN, ymax = INT_MIN, zmax = INT_MIN;
    uint8_t (*palette)[4];
    bool use_default_palette = true;
    uint8_t *voxels;
    uint8_t v[4];
    volume_iterator_t iter;
    const volume_t *volume;

    file = fopen(path, "wb");
    if (!file) {
        LOG_E("Cannot save to %s: %s", path, strerror(errno));
        return 1;
    }

    volume = goxel_get_layers_volume(image);
    palette = calloc(256, sizeof(*palette));
    for (i = 0; i < 256; i++)
        hexcolor(VOX_DEFAULT_PALETTE[i], palette[i]);

    // Iter all the voxels to get the count and the size.
    iter = volume_get_iterator(volume, VOLUME_ITER_VOXELS);
    while (volume_iter(&iter, pos)) {
        volume_get_at(volume, &iter, pos, v);
        if (v[3] < 127) continue;
        v[3] = 255;
        use_default_palette = use_default_palette &&
                            get_color_index(v, palette, true) != -1;
        nb_vox++;
        xmin = min(xmin, pos[0]);
        ymin = min(ymin, pos[1]);
        zmin = min(zmin, pos[2]);
        xmax = max(xmax, pos[0] + 1);
        ymax = max(ymax, pos[1] + 1);
        zmax = max(zmax, pos[2] + 1);
    }
    if (!use_default_palette)
        quantization_gen_palette(volume, 255, (void*)(palette + 1));

    children_size = 12 + 4 * 3 +      // SIZE chunk
                    12 + 4 + 4 * nb_vox + // XYZI chunk
                    (use_default_palette ? 0 : (12 + 4 * 256)); // RGBA chunk.

    fprintf(file, "VOX ");
    WRITE(uint32_t, 150, file);     // Version
    fprintf(file, "MAIN");
    WRITE(uint32_t, 0, file);       // Main chunck size.
    WRITE(uint32_t, children_size, file);

    fprintf(file, "SIZE");
    WRITE(uint32_t, 4 * 3, file);
    WRITE(uint32_t, 0, file);
    WRITE(uint32_t, xmax - xmin, file);
    WRITE(uint32_t, ymax - ymin, file);
    WRITE(uint32_t, zmax - zmin, file);

    fprintf(file, "XYZI");
    WRITE(uint32_t, 4 * nb_vox + 4, file);
    WRITE(uint32_t, 0, file);
    WRITE(uint32_t, nb_vox, file);

    voxels = calloc(nb_vox, 4);
    i = 0;
    iter = volume_get_iterator(volume, VOLUME_ITER_VOXELS);
    while (volume_iter(&iter, pos)) {
        volume_get_at(volume, &iter, pos, v);
        if (v[3] < 127) continue;
        pos[0] -= xmin;
        pos[1] -= ymin;
        pos[2] -= zmin;
        assert(pos[0] >= 0 && pos[0] < 255);
        assert(pos[1] >= 0 && pos[1] < 255);
        assert(pos[2] >= 0 && pos[2] < 255);

        voxels[i * 4 + 0] = pos[0];
        voxels[i * 4 + 1] = pos[1];
        voxels[i * 4 + 2] = pos[2];
        voxels[i * 4 + 3] = get_color_index(v, palette, false);
        i++;
    }
    qsort(voxels, nb_vox, 4, voxel_cmp);
    for (i = 0; i < nb_vox; i++)
        fwrite(voxels + i * 4, 4, 1, file);
    free(voxels);

    if (!use_default_palette) {
        fprintf(file, "RGBA");
        WRITE(uint32_t, 4 * 256, file);
        WRITE(uint32_t, 0, file);
        for (i = 1; i < 256; i++) {
            WRITE(uint8_t, palette[i][0], file);
            WRITE(uint8_t, palette[i][1], file);
            WRITE(uint8_t, palette[i][2], file);
            WRITE(uint8_t, palette[i][3], file);
        }
        WRITE(uint32_t, 0, file);
    }

    fclose(file);
    free(palette);
    return 0;
}

FILE_FORMAT_REGISTER(vox,
    .name = "Magica Voxel",
    .exts = {"*.vox"},
    .exts_desc = "vox",
    .import_func = vox_import,
    .export_func = vox_export,
)


static const uint32_t VOX_DEFAULT_PALETTE[256] = {
    0x00000000, 0xffffffff, 0xffffccff, 0xffff99ff, 0xffff66ff, 0xffff33ff,
    0xffff00ff, 0xffccffff, 0xffccccff, 0xffcc99ff, 0xffcc66ff, 0xffcc33ff,
    0xffcc00ff, 0xff99ffff, 0xff99ccff, 0xff9999ff, 0xff9966ff, 0xff9933ff,
    0xff9900ff, 0xff66ffff, 0xff66ccff, 0xff6699ff, 0xff6666ff, 0xff6633ff,
    0xff6600ff, 0xff33ffff, 0xff33ccff, 0xff3399ff, 0xff3366ff, 0xff3333ff,
    0xff3300ff, 0xff00ffff, 0xff00ccff, 0xff0099ff, 0xff0066ff, 0xff0033ff,
    0xff0000ff, 0xccffffff, 0xccffccff, 0xccff99ff, 0xccff66ff, 0xccff33ff,
    0xccff00ff, 0xccccffff, 0xccccccff, 0xcccc99ff, 0xcccc66ff, 0xcccc33ff,
    0xcccc00ff, 0xcc99ffff, 0xcc99ccff, 0xcc9999ff, 0xcc9966ff, 0xcc9933ff,
    0xcc9900ff, 0xcc66ffff, 0xcc66ccff, 0xcc6699ff, 0xcc6666ff, 0xcc6633ff,
    0xcc6600ff, 0xcc33ffff, 0xcc33ccff, 0xcc3399ff, 0xcc3366ff, 0xcc3333ff,
    0xcc3300ff, 0xcc00ffff, 0xcc00ccff, 0xcc0099ff, 0xcc0066ff, 0xcc0033ff,
    0xcc0000ff, 0x99ffffff, 0x99ffccff, 0x99ff99ff, 0x99ff66ff, 0x99ff33ff,
    0x99ff00ff, 0x99ccffff, 0x99ccccff, 0x99cc99ff, 0x99cc66ff, 0x99cc33ff,
    0x99cc00ff, 0x9999ffff, 0x9999ccff, 0x999999ff, 0x999966ff, 0x999933ff,
    0x999900ff, 0x9966ffff, 0x9966ccff, 0x996699ff, 0x996666ff, 0x996633ff,
    0x996600ff, 0x9933ffff, 0x9933ccff, 0x993399ff, 0x993366ff, 0x993333ff,
    0x993300ff, 0x9900ffff, 0x9900ccff, 0x990099ff, 0x990066ff, 0x990033ff,
    0x990000ff, 0x66ffffff, 0x66ffccff, 0x66ff99ff, 0x66ff66ff, 0x66ff33ff,
    0x66ff00ff, 0x66ccffff, 0x66ccccff, 0x66cc99ff, 0x66cc66ff, 0x66cc33ff,
    0x66cc00ff, 0x6699ffff, 0x6699ccff, 0x669999ff, 0x669966ff, 0x669933ff,
    0x669900ff, 0x6666ffff, 0x6666ccff, 0x666699ff, 0x666666ff, 0x666633ff,
    0x666600ff, 0x6633ffff, 0x6633ccff, 0x663399ff, 0x663366ff, 0x663333ff,
    0x663300ff, 0x6600ffff, 0x6600ccff, 0x660099ff, 0x660066ff, 0x660033ff,
    0x660000ff, 0x33ffffff, 0x33ffccff, 0x33ff99ff, 0x33ff66ff, 0x33ff33ff,
    0x33ff00ff, 0x33ccffff, 0x33ccccff, 0x33cc99ff, 0x33cc66ff, 0x33cc33ff,
    0x33cc00ff, 0x3399ffff, 0x3399ccff, 0x339999ff, 0x339966ff, 0x339933ff,
    0x339900ff, 0x3366ffff, 0x3366ccff, 0x336699ff, 0x336666ff, 0x336633ff,
    0x336600ff, 0x3333ffff, 0x3333ccff, 0x333399ff, 0x333366ff, 0x333333ff,
    0x333300ff, 0x3300ffff, 0x3300ccff, 0x330099ff, 0x330066ff, 0x330033ff,
    0x330000ff, 0x00ffffff, 0x00ffccff, 0x00ff99ff, 0x00ff66ff, 0x00ff33ff,
    0x00ff00ff, 0x00ccffff, 0x00ccccff, 0x00cc99ff, 0x00cc66ff, 0x00cc33ff,
    0x00cc00ff, 0x0099ffff, 0x0099ccff, 0x009999ff, 0x009966ff, 0x009933ff,
    0x009900ff, 0x0066ffff, 0x0066ccff, 0x006699ff, 0x006666ff, 0x006633ff,
    0x006600ff, 0x0033ffff, 0x0033ccff, 0x003399ff, 0x003366ff, 0x003333ff,
    0x003300ff, 0x0000ffff, 0x0000ccff, 0x000099ff, 0x000066ff, 0x000033ff,
    0xee0000ff, 0xdd0000ff, 0xbb0000ff, 0xaa0000ff, 0x880000ff, 0x770000ff,
    0x550000ff, 0x440000ff, 0x220000ff, 0x110000ff, 0x00ee00ff, 0x00dd00ff,
    0x00bb00ff, 0x00aa00ff, 0x008800ff, 0x007700ff, 0x005500ff, 0x004400ff,
    0x002200ff, 0x001100ff, 0x0000eeff, 0x0000ddff, 0x0000bbff, 0x0000aaff,
    0x000088ff, 0x000077ff, 0x000055ff, 0x000044ff, 0x000022ff, 0x000011ff,
    0xeeeeeeff, 0xddddddff, 0xbbbbbbff, 0xaaaaaaff, 0x888888ff, 0x777777ff,
    0x555555ff, 0x444444ff, 0x222222ff, 0x111111ff,
};
