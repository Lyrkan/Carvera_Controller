// Tool position pointer mesh (pointer.obj) with simple diffuse shading

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

void main()
{
    float shade = abs(dot(normal_vec.xyz, vec3(1.0, 1.0, 1.0)));
    vec3 color = vec3(0.3, 0.3, 1.0) * shade;
    gl_FragColor = vec4(color, 0.3) * texture2D(texture0, tex_coord0);
}
