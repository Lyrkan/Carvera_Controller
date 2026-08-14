// Carved voxel chunk surface (lit opaque fill with height tint).

---vertex
$HEADER$

attribute vec3 v_pos;
attribute vec3 v_normal;
attribute vec4 v_color;

uniform mat4 center_offset;
uniform mat4 view_mat;
uniform mat4 proj_mat;
uniform mat4 rotation_mat;
uniform float vertex_scale;

varying vec3 normal_vec;
varying vec4 voxel_color;
varying float world_z;

void main()
{
    vec3 scaled_pos = v_pos * vertex_scale;
    vec4 rotated = rotation_mat * vec4(scaled_pos, 1.0);
    vec4 world = center_offset * rotated;
    vec4 eye_pos = view_mat * world;
    normal_vec = (view_mat * rotation_mat * vec4(v_normal, 0.0)).xyz;
    voxel_color = v_color;
    // Height tint after A rotation so "up" stays machine Z, not stock-local Z.
    world_z = rotated.z;
    tex_coord0 = vec2(0.0);
    gl_Position = proj_mat * eye_pos;
}

---fragment
$HEADER$

varying vec3 normal_vec;
varying vec4 voxel_color;
varying float world_z;

uniform float stock_z_min;
uniform float stock_z_max;

void main()
{
    vec3 n = normalize(normal_vec);
    // Stronger contrast than the translucent AABB stock shader so pocket walls read.
    float ndl = abs(dot(n, normalize(vec3(0.35, 0.55, 1.0))));
    float shade = 0.18 + 0.82 * ndl;

    // Darker toward stock bottom so coplanar pocket floors separate by depth.
    float z_span = max(stock_z_max - stock_z_min, 1e-6);
    float t = clamp((world_z - stock_z_min) / z_span, 0.0, 1.0);
    vec3 low = voxel_color.rgb * vec3(0.52, 0.48, 0.42);
    vec3 high = voxel_color.rgb * vec3(1.08, 1.05, 0.98);
    vec3 tinted = mix(low, high, t);

    vec3 color = tinted * shade;
    gl_FragColor = vec4(color, voxel_color.a) * texture2D(texture0, tex_coord0);
}
