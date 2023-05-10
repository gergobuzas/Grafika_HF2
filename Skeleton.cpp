//=============================================================================================
// Mintaprogram: Z�ld h�romsz�g. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Buzas Gergo
// Neptun : E0PWAX
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

struct Material {
    vec3 ka, kd, ks; //ambient, diffuse, specular
    float  shininess;
    Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd * M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
    float t; //ray parameter
    vec3 position;
    vec3 normal;
    Material* material;
    Hit() { t = -1; }
};

class Intersectable {
    protected:
        Material * material;
    public:
        virtual Hit intersect(const Ray& ray) = 0;
};

class Triangle : public Intersectable{
    vec3 a, b, c, normalVec;
    public:
        Triangle(vec3 _a, vec3 _b, vec3 _c, Material* _material) {
            a = _a; b = _b; c = _c; material = _material;
            normalVec = normalize(cross((b - a), (c - a)));
        }
        Hit intersect(const Ray& ray) {
            Hit hit;
            vec3 n = cross(b - a, c - a);
            n = normalize(n);
            float t = dot(a - ray.start, n) / dot(ray.dir, n);  //ray parameter
            if (t < 0) return hit;
            vec3 p = ray.start + ray.dir * t; //intersection point
            //vec3 sikmetszes = (p - a) * n;
            float cond1 = dot(cross(b - a,p - a), n);
            float cond2 = dot(cross(c - b,p - b), n);
            float cond3 = dot(cross(a - c,p - c), n);
            if (cond1 >= 0  && cond2 >= 0 && cond3 >= 0){
                hit.t = t;
                hit.position = p;
                hit.normal = n;
                hit.material = material;
                return hit;
            }
            return hit;
        }
};


class Cube : public Intersectable{

};

class Cone : public Intersectable {


};

class Octahedron : public Intersectable {

};

class Icosahedron : public Intersectable {

};



class Camera {
    vec3 eye, lookat, right, up;
    public:
        void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
            eye = _eye;
            lookat = _lookat;
            vec3 w = eye - lookat;
            float focus = length(w);
            right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
            up = normalize(cross(w, right)) * focus * tanf(fov / 2);
        }
        Ray getRay(int X, int Y) {
            vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
            return Ray(eye, dir);
        }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

void drawCube(std::vector<Intersectable*>& objects, Material* material){
    //TODO create a cube with 12 triangles (2 per side) and add it to the list of objects (objects.push_back(...))
    vec3 a = vec3(0, 0, 0);             //1
    vec3 b = vec3(0, 0, 1.0f);          //2
    vec3 c = vec3(0, 1.0f, 0);          //3
    vec3 d = vec3(0, 1.0f, 1.0f);       //4
    vec3 e = vec3(1.0f, 0, 0);          //5
    vec3 f = vec3(1.0f, 0, 1.0f);       //6
    vec3 g = vec3(1.0f, 1.0f, 0);       //7
    vec3 h = vec3(1.0f, 1.0f, 1.0f);    //8
    objects.push_back(new Triangle(a, g, e, material));
    objects.push_back(new Triangle(a, c, g, material));
    objects.push_back(new Triangle(a, d, c, material));
    objects.push_back(new Triangle(a, b, d, material));
    objects.push_back(new Triangle(c, h, g, material));
    objects.push_back(new Triangle(c, d, h, material));
    objects.push_back(new Triangle(e, g, h, material));
    objects.push_back(new Triangle(e, h, f, material));
    objects.push_back(new Triangle(a, e, f, material));
    objects.push_back(new Triangle(a, f, b, material));
    objects.push_back(new Triangle(b, f, h, material));
    objects.push_back(new Triangle(b, h, d, material));
}

void drawOctahedron(std::vector<Intersectable*>& objects, Material* material){
    vec3 transform = vec3(0.2f, 0.7f, 0);
    vec3 a = vec3(0.2f, 0, 0) + transform;             //1
    vec3 b = vec3(0, -0.2f, 0) + transform;          //2
    vec3 c = vec3(-0.2f, 0, 0) + transform;          //3
    vec3 d = vec3(0, 0.2f, 0) + transform;       //4
    vec3 e = vec3(0, 0, 0.2f) + transform;          //5
    vec3 f = vec3(0, 0, -0.2f) + transform;       //6
    objects.push_back(new Triangle(b, a, e, material));
    objects.push_back(new Triangle(c, b, e, material));
    objects.push_back(new Triangle(d, c, e, material));
    objects.push_back(new Triangle(a, d, e, material));
    objects.push_back(new Triangle(a, b, f, material));
    objects.push_back(new Triangle(b, c, f, material));
    objects.push_back(new Triangle(c, d, f, material));
    objects.push_back(new Triangle(d, a, f, material));
}

void drawIcosahedron(std::vector<Intersectable*>& objects, Material* material){
    //TODO create an icosahedron with triangles and add it to the list of objects (objects.push_back(...))
}

void drawCone(std::vector<Intersectable*>& objects, Material* material){
    //TODO create a cone with triangles and add it to the list of objects (objects.push_back(...))
}

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La;
public:
    void build() {
        vec3 eye = vec3(1.9f, 0.83f, 2.2f), vup = vec3(0, 1.0f, 0), lookat = vec3(0, 0.2f, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.2f, 0.2f, 0.2f);
        vec3 lightDirection(-1, 1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        vec3 kd(0.5f, 0.5f, 0.5f), ks(2, 2, 2);
        Material* material = new Material(kd, ks, 50);
        drawCube(objects, material);
        drawOctahedron(objects, material);


    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
            for (int X = 0; X < windowWidth; X++) {
                Ray ray = camera.getRay(X, Y);
                vec3 color = trace(ray);
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))
                bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0)
            bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    static std::vector<Hit> orderHits(std::vector<Hit>& hits){
        //Order hits by distance
        for (int i = 0; i < hits.size(); i++) {
            for (int j = i; j < hits.size(); j++) {
                if (hits[i].t > hits[j].t) {
                    Hit temp = hits[i];
                    hits[i] = hits[j];
                    hits[j] = temp;
                }
            }
        }
        return hits;
    }

    Hit secondIntersect(Ray ray){
        std::vector<Hit> hits = std::vector<Hit>();
        for (Intersectable * object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0)
                hits.push_back(hit);
        }
        if (hits.empty())
            return Hit();
        if (hits.size() == 1)
            return hits[0];
        hits = orderHits(hits);
        if (dot(ray.dir, hits[1].normal) > 0)
            hits[1].normal = hits[1].normal * (-1);
        return hits[1];
    }

    bool shadowIntersect(Ray ray) {	// for directional lights
        for (Intersectable * object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = secondIntersect(ray);
        if (hit.t < 0) return vec3(0,0,0);
        float La = 0.2 * (1 + dot(hit.normal, ray.dir * (-1)));
        vec3 outRadiance = vec3(La, La, La);
        for (Light* light : lights) {
            Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
            float cosTheta = dot(hit.normal, light->direction);
            if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
                outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                vec3 halfway = normalize(-ray.dir + light->direction);
                float cosDelta = dot(hit.normal, halfway);
                if (cosDelta > 0) outRadiance = outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
            }
        }
        return outRadiance;
    }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

class FullScreenTexturedQuad {
    unsigned int vao;	// vertex array object id and texture id
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
            : texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);	// create 1 vertex array object
        glBindVertexArray(vao);		// make it active

        unsigned int vbo;		// vertex buffer objects
        glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
    }

    void Draw() {
        glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
    }
};

FullScreenTexturedQuad * fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);
    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}

