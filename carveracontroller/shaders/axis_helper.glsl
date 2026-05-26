// XYZ axis arrow meshes (axis.obj) with diffuse shading and per-axis color

---vertex
$HEADER$

attribute vec3 v_pos;
attribute vec3 v_normal;

uniform vec3 offset;
uniform mat4 rotation;

varying vec4 normal_vec;

void main()
{
    vec4 rotated = rotation * vec4(v_pos, 1.0);
    vec3 world_pos = rotated.xyz + offset;
    vec4 eye_pos = modelview_mat * vec4(world_pos, 1.0);
    normal_vec = vec4(v_normal, 0.0);
    tex_coord0 = vec2(0.0);
    gl_Position = projection_mat * eye_pos;
}

---fragment
$HEADER$

varying vec4 normal_vec;

uniform vec3 diff_color;

void main()
{
    float shade = abs(dot(normal_vec.xyz, vec3(1.0, 1.0, 1.0)));
    vec3 color = diff_color * shade;
    gl_FragColor = vec4(color, 1.0) * texture2D(texture0, tex_coord0);
}
