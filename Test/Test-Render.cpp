#include "mnist_reader.hpp"
#include "NN.hpp"
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

using namespace std;
GLuint create_shader(const char *shader_path, GLenum shader_type){
    ifstream filestream(shader_path,ios::in);
    if(!filestream.is_open()){
        cerr << "Error opening file named: " << shader_path << '\n';
        return -1;
    }
    
    ostringstream ss;
    ss << filestream.rdbuf();
    string str = ss.str();
    const char *src = str.c_str();
    
    GLuint shader = glCreateShader(shader_type);
    glShaderSource(shader,1,&src,NULL);
    glCompileShader(shader);
    
    int compile_status;
    glGetShaderiv(shader,GL_COMPILE_STATUS,&compile_status);

    if(!compile_status){
        int log_length;
        glGetShaderiv(shader,GL_INFO_LOG_LENGTH,&log_length);
        char log[log_length];
        glGetShaderInfoLog(shader,log_length,NULL,log);
        cerr << log << '\n';
        return -1;
    }
    filestream.close();

    return shader;
}   

GLuint create_program(const char *vs_shader_path, const char *fs_shader_path){
    GLuint program_id = glCreateProgram();
    glAttachShader(program_id,create_shader(vs_shader_path,GL_VERTEX_SHADER));
    glAttachShader(program_id,create_shader(fs_shader_path,GL_FRAGMENT_SHADER));
    glLinkProgram(program_id);

    int link_status;
    glGetProgramiv(program_id,GL_LINK_STATUS,&link_status);

    if(!link_status){
        int log_length;
        glGetProgramiv(program_id,GL_INFO_LOG_LENGTH,&log_length);
        char log[log_length];
        glGetProgramInfoLog(program_id,log_length,NULL,log);
        cerr << log << '\n';
        return -1;
    }
    return program_id;
}

int main(){
    SDL_Init(SDL_INIT_VIDEO);

    SDL_DisplayMode dm;
    SDL_GetDesktopDisplayMode(0,&dm);

    const int window_width = dm.w, window_height = dm.h, frame_latency = 1000000/dm.refresh_rate,rows = 28, cols = 28;

    SDL_Window *window = SDL_CreateWindow("Autoencoder Test",SDL_WINDOWPOS_UNDEFINED,SDL_WINDOWPOS_UNDEFINED,window_width,window_height,SDL_WINDOW_OPENGL|SDL_WINDOW_FULLSCREEN_DESKTOP);
    SDL_GLContext context = SDL_GL_CreateContext(window);

    glewExperimental = GL_TRUE;
    glewInit();

    int sample_size = 500;
    mnist_reader<float> reader(sample_size,"Test/Dataset/train-images","Test/Dataset/train-labels");

    SDL_Event event;
    bool quit = false;

    GLuint vbo,vao,ebo,vbo_offsets,vbo_color;
    {
        float offsets[rows*cols*2],vertices[] = {
            -1,1,
            -1+2.0/cols,1,
            -1+2.0/cols,1-2.0/rows,
            -1,1-2.0/rows
        };
        unsigned int indices[] = {
            0,1,2,
            2,3,0   
        };

        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < cols; ++j){
                offsets[i*cols*2+j*2] = 2.0*j/cols;
                offsets[i*cols*2+j*2+1] = -2.0*i/rows;
            }
        }

        glGenBuffers(1,&vbo);
        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glBufferData(GL_ARRAY_BUFFER,sizeof(vertices),vertices,GL_STATIC_DRAW);

        glGenBuffers(1,&ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER,sizeof(indices),indices,GL_STATIC_DRAW);

        glGenBuffers(1,&vbo_offsets);
        glBindBuffer(GL_ARRAY_BUFFER,vbo_offsets);
        glBufferData(GL_ARRAY_BUFFER,sizeof(offsets),offsets,GL_STATIC_DRAW);

        glGenBuffers(1,&vbo_color);
        glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
        glBufferData(GL_ARRAY_BUFFER,sizeof(float)*rows*cols,reader.get_processed_image(),GL_DYNAMIC_DRAW);

        glGenVertexArrays(1,&vao);
        glBindVertexArray(vao);

        glBindBuffer(GL_ARRAY_BUFFER,vbo);
        glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,(void*)0);
        glEnableVertexAttribArray(0);

        glBindBuffer(GL_ARRAY_BUFFER,vbo_offsets);
        glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE,sizeof(float)*2,(void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribDivisor(1,1);

        glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
        glVertexAttribPointer(2,1,GL_FLOAT,GL_FALSE,sizeof(float),(void*)0);
        glEnableVertexAttribArray(2);
        glVertexAttribDivisor(2,1);
    };

    GLuint program = create_program("Test/Shader/vs.glsl","Test/Shader/fs.glsl");

    chrono::time_point<chrono::high_resolution_clock> reference = chrono::high_resolution_clock::now();
    int index = 0;
    bool last_key = false;

    int batch_size = 50;
    neural_network<float> net(batch_size,5);
    net.add_input_layer(784,false);
    net.add_fully_connected_layer(200,TANH);
    net.add_fully_connected_layer(10,TANH);
    net.add_fully_connected_layer(200,TANH);
    net.add_fully_connected_layer(784,TANH);
    net.construct_net();

    int indices[sample_size/batch_size];
    iota(indices,indices+sample_size/batch_size,0);
    while(!quit){
        while(SDL_PollEvent(&event)){
            if(event.type == SDL_QUIT){
                quit = true;
                break;
            }else if(event.type == SDL_KEYDOWN){
                switch(event.key.keysym.sym){
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_w:
                        if(last_key) continue;
                        last_key = true;
                        if(index < sample_size-1) ++index;
                        break;
                    case SDLK_s:
                        if(last_key) continue;
                        last_key = true;
                        if(index > 0) --index;
                        break;
                }
            }else if(event.type == SDL_KEYUP){
                last_key = false;
            }
        }

        if(chrono::duration_cast<chrono::microseconds>(chrono::high_resolution_clock::now()-reference).count() >= frame_latency){
            glClearColor(0.2f,0.2f,0.2f,1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            glViewport(0,0,window_width/2,window_height);

            glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
            glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*rows*cols,reader.get_processed_image()+784*index);

            glUseProgram(program);
            glBindVertexArray(vao);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
            glDrawElementsInstanced(GL_TRIANGLES,6,GL_UNSIGNED_INT,(void*)0,rows*cols);

            glViewport(window_width/2,0,window_width/2,window_height);
            float *net_out = net.feedforward(reader.get_processed_image_mean_normalized()+index/batch_size*batch_size);
            
            random_shuffle(indices,indices+sample_size/batch_size);
            for(int i = 0; i < sample_size/batch_size; ++i) net.backpropagate(reader.get_processed_image_mean_normalized()+784*batch_size*indices[i],reader.get_processed_image_mean_normalized()+784*batch_size*indices[i]);
            
            for(int i = 0; i < 784*batch_size; ++i) net_out[i] = net_out[i]*0.5+0.5;
            cout << "loss: " << net.get_loss(reader.get_processed_image_mean_normalized(),reader.get_processed_image_mean_normalized()) << '\n';
            glBindBuffer(GL_ARRAY_BUFFER,vbo_color);
            glBufferSubData(GL_ARRAY_BUFFER,0,sizeof(float)*rows*cols,net_out+(index%batch_size)*784);

            glUseProgram(program);
            glBindVertexArray(vao);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,ebo);
            glDrawElementsInstanced(GL_TRIANGLES,6,GL_UNSIGNED_INT,(void*)0,rows*cols);

            SDL_GL_SwapWindow(window);
            reference = chrono::high_resolution_clock::now();
        }
    }

    SDL_GL_DeleteContext(context);
    SDL_Quit();

    return 0;
}