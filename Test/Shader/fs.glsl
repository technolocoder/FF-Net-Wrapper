#version 420 core

in VS_OUT{
    float color;
} fs_in;

out vec4 color;

void main(){
color = vec4(vec3(fs_in.color),1.0);
}