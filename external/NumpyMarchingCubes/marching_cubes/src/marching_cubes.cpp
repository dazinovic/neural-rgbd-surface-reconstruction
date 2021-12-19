#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <list>

#include "tables.h"
#include "sparsegrid3.h"
#include "marching_cubes.h"

#define VOXELSIZE 1.0f

struct vec3f {
	vec3f() {
		x = 0.0f;
		y = 0.0f;
		z = 0.0f;
	}
	vec3f(float x_, float y_, float z_) {
		x = x_;
		y = y_;
		z = z_;
	}
	inline vec3f operator+(const vec3f& other) const {
		return vec3f(x+other.x, y+other.y, z+other.z);
	}
	inline vec3f operator-(const vec3f& other) const {
		return vec3f(x-other.x, y-other.y, z-other.z);
	}
	inline vec3f operator*(float val) const {
		return vec3f(x*val, y*val, z*val);
	}
	inline void operator+=(const vec3f& other) {
		x += other.x;
		y += other.y;
		z += other.z;
	}
	static float distSq(const vec3f& v0, const vec3f& v1) {
		return ((v0.x-v1.x)*(v0.x-v1.x) + (v0.y-v1.y)*(v0.y-v1.y) + (v0.z-v1.z)*(v0.z-v1.z));
	}
	float x;
	float y;
	float z;
};
inline vec3f operator*(float s, const vec3f& v) {
	return v * s;
}
struct vec3uc {
	vec3uc() {
		x = 0;
		y = 0;
		z = 0;
	}
	vec3uc(unsigned char x_, unsigned char y_, unsigned char z_) {
		x = x_;
		y = y_;
		z = z_;
	}
	unsigned char x;
	unsigned char y;
	unsigned char z;
};

struct Triangle {
	vec3f v0;
	vec3f v1;
	vec3f v2;
};

void get_voxel(
	const vec3f& pos,
	const npy_accessor& tsdf_accessor,
	float truncation, 
	float& d, 
	int& w) {
	int x = (int)round(pos.x);
	int y = (int)round(pos.y);
	int z = (int)round(pos.z);
	if (z >= 0 && z < tsdf_accessor.size()[2] &&
		y >= 0 && y < tsdf_accessor.size()[1] &&
		x >= 0 && x < tsdf_accessor.size()[0]) {
		d = tsdf_accessor(x, y, z);
		if (d != -std::numeric_limits<float>::infinity() && fabs(d) < truncation) w = 1;
		else w = 0;
	}
	else {
		d = -std::numeric_limits<float>::infinity();
		w = 0;
	}
}

bool trilerp(
	const vec3f& pos, 
	float& dist,
	const npy_accessor& tsdf_accessor,
	float truncation)  {
	const float oSet = VOXELSIZE;
	const vec3f posDual = pos - vec3f(oSet / 2.0f, oSet / 2.0f, oSet / 2.0f);
	vec3f weight = vec3f(pos.x - (int)pos.x, pos.y - (int)pos.y, pos.z - (int)pos.z);

	dist = 0.0f;
	float d; int w;
	get_voxel(posDual + vec3f(0.0f, 0.0f, 0.0f), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += (1.0f - weight.x)*(1.0f - weight.y)*(1.0f - weight.z)*d;
	get_voxel(posDual + vec3f(oSet, 0.0f, 0.0f), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += weight.x *(1.0f - weight.y)*(1.0f - weight.z)*d;
	get_voxel(posDual + vec3f(0.0f, oSet, 0.0f), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += (1.0f - weight.x)*	   weight.y *(1.0f - weight.z)*d;
	get_voxel(posDual + vec3f(0.0f, 0.0f, oSet), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += (1.0f - weight.x)*(1.0f - weight.y)*	   weight.z *d;
	get_voxel(posDual + vec3f(oSet, oSet, 0.0f), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += weight.x *	   weight.y *(1.0f - weight.z)*d;
	get_voxel(posDual + vec3f(0.0f, oSet, oSet), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += (1.0f - weight.x)*	   weight.y *	   weight.z *d;
	get_voxel(posDual + vec3f(oSet, 0.0f, oSet), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += weight.x *(1.0f - weight.y)*	   weight.z *d;
	get_voxel(posDual + vec3f(oSet, oSet, oSet), tsdf_accessor, truncation, d, w); if (w == 0) return false; dist += weight.x *	   weight.y *	   weight.z *d;

	return true;
}

vec3f vertexInterp(float isolevel, const vec3f& p1, const vec3f& p2, float d1, float d2)
{
	vec3f r1 = p1;
	vec3f r2 = p2;
    //printf("[interp] r1 = (%f, %f, %f), r2 = (%f, %f, %f) d1 = %f, d2 = %f, iso = %f\n", r1.x, r1.y, r1.z, r2.x, r2.y, r2.z, d1, d2, isolevel);
	//printf("%d, %d, %d || %f, %f, %f -> %f, %f, %f\n", fabs(isolevel - d1) < 0.00001f, fabs(isolevel - d2) < 0.00001f, fabs(d1 - d2) < 0.00001f, isolevel - d1, isolevel - d2, d1-d2, fabs(isolevel - d1), fabs(isolevel - d2), fabs(d1-d2));

	if (fabs(isolevel - d1) < 0.00001f)		return r1;
	if (fabs(isolevel - d2) < 0.00001f)		return r2;
	if (fabs(d1 - d2) < 0.00001f)			return r1;

	float mu = (isolevel - d1) / (d2 - d1);

	vec3f res;
	res.x = p1.x + mu * (p2.x - p1.x); // Positions
	res.y = p1.y + mu * (p2.y - p1.y);
	res.z = p1.z + mu * (p2.z - p1.z);
	
	//printf("[interp] mu = %f, res = (%f, %f, %f)     r1 = (%f, %f, %f), r2 = (%f, %f, %f)\n", mu, res.x, res.y, res.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z);

	return res;
}

void extract_isosurface_at_position(
    const vec3f& pos,
	const npy_accessor& tsdf_accessor,
	float truncation,
	float isolevel,
	float thresh,
	std::vector<Triangle>& results) {
	const float voxelsize = VOXELSIZE;
	const float P = voxelsize / 2.0f;
	const float M = -P;

    //const bool debugprint = (pos.z == 33 && pos.y == 56 && pos.x == 2) || (pos.z == 2 && pos.y == 56 && pos.x == 33);

	vec3f p000 = pos + vec3f(M, M, M); float dist000; bool valid000 = trilerp(p000, dist000, tsdf_accessor, truncation);
	vec3f p100 = pos + vec3f(P, M, M); float dist100; bool valid100 = trilerp(p100, dist100, tsdf_accessor, truncation);
	vec3f p010 = pos + vec3f(M, P, M); float dist010; bool valid010 = trilerp(p010, dist010, tsdf_accessor, truncation);
	vec3f p001 = pos + vec3f(M, M, P); float dist001; bool valid001 = trilerp(p001, dist001, tsdf_accessor, truncation);
	vec3f p110 = pos + vec3f(P, P, M); float dist110; bool valid110 = trilerp(p110, dist110, tsdf_accessor, truncation);
	vec3f p011 = pos + vec3f(M, P, P); float dist011; bool valid011 = trilerp(p011, dist011, tsdf_accessor, truncation);
	vec3f p101 = pos + vec3f(P, M, P); float dist101; bool valid101 = trilerp(p101, dist101, tsdf_accessor, truncation);
	vec3f p111 = pos + vec3f(P, P, P); float dist111; bool valid111 = trilerp(p111, dist111, tsdf_accessor, truncation);
	//if (debugprint) {
	//	printf("[extract_isosurface_at_position] pos: %f, %f, %f\n", pos.x, pos.y, pos.z);
	//	printf("\tp000 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p000.x, p000.y, p000.z, dist000, (int)color000.x, (int)color000.y, (int)color000.z, (int)valid000);
	//	printf("\tp100 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p100.x, p100.y, p100.z, dist100, (int)color100.x, (int)color100.y, (int)color100.z, (int)valid100);
	//	printf("\tp010 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p010.x, p010.y, p010.z, dist010, (int)color010.x, (int)color010.y, (int)color010.z, (int)valid010);
	//	printf("\tp001 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p001.x, p001.y, p001.z, dist001, (int)color001.x, (int)color001.y, (int)color001.z, (int)valid001);
	//	printf("\tp110 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p110.x, p110.y, p110.z, dist110, (int)color110.x, (int)color110.y, (int)color110.z, (int)valid110);
	//	printf("\tp011 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p011.x, p011.y, p011.z, dist011, (int)color011.x, (int)color011.y, (int)color011.z, (int)valid011);
	//	printf("\tp101 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p101.x, p101.y, p101.z, dist101, (int)color101.x, (int)color101.y, (int)color101.z, (int)valid101);
	//	printf("\tp111 (%f, %f, %f) -> dist %f, color %d, %d, %d | valid %d\n", p111.x, p111.y, p111.z, dist111, (int)color111.x, (int)color111.y, (int)color111.z, (int)valid111);
	//}
	if (!valid000 || !valid100 || !valid010 || !valid001 || !valid110 || !valid011 || !valid101 || !valid111) return;
	
	uint cubeindex = 0;
	if (dist010 < isolevel) cubeindex += 1;
	if (dist110 < isolevel) cubeindex += 2;
	if (dist100 < isolevel) cubeindex += 4;
	if (dist000 < isolevel) cubeindex += 8;
	if (dist011 < isolevel) cubeindex += 16;
	if (dist111 < isolevel) cubeindex += 32;
	if (dist101 < isolevel) cubeindex += 64;
	if (dist001 < isolevel) cubeindex += 128;
	const float thres = thresh;
	float distArray[] = { dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111 };
	//if (debugprint) {
	//	printf("dists (%f, %f, %f, %f, %f, %f, %f, %f)\n", dist000, dist100, dist010, dist001, dist110, dist011, dist101, dist111);
	//	printf("cubeindex %d\n", cubeindex);
	//}
	for (uint k = 0; k < 8; k++) {
		for (uint l = 0; l < 8; l++) {
			if (distArray[k] * distArray[l] < 0.0f) {
				if (fabs(distArray[k]) + fabs(distArray[l]) > thres) return;
			}
			else {
				if (fabs(distArray[k] - distArray[l]) > thres) return;
			}
		}
	}
	if (fabs(dist000) > thresh) return;
	if (fabs(dist100) > thresh) return;
	if (fabs(dist010) > thresh) return;
	if (fabs(dist001) > thresh) return;
	if (fabs(dist110) > thresh) return;
	if (fabs(dist011) > thresh) return;
	if (fabs(dist101) > thresh) return;
	if (fabs(dist111) > thresh) return;
	
	if (edgeTable[cubeindex] == 0 || edgeTable[cubeindex] == 255) return; // added by me edgeTable[cubeindex] == 255

	vec3uc c;
	{
		float d; int w; 
		get_voxel(pos, tsdf_accessor, truncation, d, w);
	}
	
	vec3f vertlist[12];
	if (edgeTable[cubeindex] & 1)	    vertlist[0] = vertexInterp(isolevel, p010, p110, dist010, dist110);
	if (edgeTable[cubeindex] & 2)	    vertlist[1] = vertexInterp(isolevel, p110, p100, dist110, dist100);
	if (edgeTable[cubeindex] & 4)	    vertlist[2] = vertexInterp(isolevel, p100, p000, dist100, dist000);
	if (edgeTable[cubeindex] & 8)	    vertlist[3] = vertexInterp(isolevel, p000, p010, dist000, dist010);
	if (edgeTable[cubeindex] & 16)	vertlist[4] = vertexInterp(isolevel, p011, p111, dist011, dist111);
	if (edgeTable[cubeindex] & 32)	vertlist[5] = vertexInterp(isolevel, p111, p101, dist111, dist101);
	if (edgeTable[cubeindex] & 64)	vertlist[6] = vertexInterp(isolevel, p101, p001, dist101, dist001);
	if (edgeTable[cubeindex] & 128)	vertlist[7] = vertexInterp(isolevel, p001, p011, dist001, dist011);
	if (edgeTable[cubeindex] & 256)	vertlist[8] = vertexInterp(isolevel, p010, p011, dist010, dist011);
	if (edgeTable[cubeindex] & 512)	vertlist[9] = vertexInterp(isolevel, p110, p111, dist110, dist111);
	if (edgeTable[cubeindex] & 1024)  vertlist[10] = vertexInterp(isolevel, p100, p101, dist100, dist101);
	if (edgeTable[cubeindex] & 2048)  vertlist[11] = vertexInterp(isolevel, p000, p001, dist000, dist001);

	for (int i = 0; triTable[cubeindex][i] != -1; i += 3)
	{
		Triangle t;
		t.v0 = vertlist[triTable[cubeindex][i + 0]];
		t.v1 = vertlist[triTable[cubeindex][i + 1]];
		t.v2 = vertlist[triTable[cubeindex][i + 2]];

        //printf("triangle at (%f, %f, %f): (%f, %f, %f) (%f, %f, %f) (%f, %f, %f)\n", pos.x, pos.y, pos.z, t.v0.x, t.v0.y, t.v0.z, t.v1.x, t.v1.y, t.v1.z, t.v2.x, t.v2.y, t.v2.z);
		//printf("vertlist idxs: %d, %d, %d (%d, %d, %d)\n", triTable[cubeindex][i + 0], triTable[cubeindex][i + 1], triTable[cubeindex][i + 2], edgeTable[cubeindex] & 1, edgeTable[cubeindex] & 256, edgeTable[cubeindex] & 8);
		//getchar();
		results.push_back(t);
	}
}


// ----- MESH CLEANUP FUNCTIONS
unsigned int remove_duplicate_faces(std::vector<vec3i>& faces)
{
	struct vecHash {
		size_t operator()(const std::vector<unsigned int>& v) const {
			//TODO larger prime number (64 bit) to match size_t
			const size_t p[] = {73856093, 19349669, 83492791};
			size_t res = 0;
			for (unsigned int i : v) {
				res = res ^ (size_t)i * p[i%3];
			}
			return res;
			//const size_t res = ((size_t)v.x * p0)^((size_t)v.y * p1)^((size_t)v.z * p2);
		}
	};

	size_t numFaces = faces.size();
	std::vector<vec3i> new_faces;	new_faces.reserve(numFaces);

	std::unordered_set<std::vector<unsigned int>, vecHash> _set;
	for (size_t i = 0; i < numFaces; i++) {
		std::vector<unsigned int> face = {(unsigned int)faces[i].x, (unsigned int)faces[i].y, (unsigned int)faces[i].z};
		std::sort(face.begin(), face.end());
		if (_set.find(face) == _set.end()) {
			//not found yet
			_set.insert(face);
			new_faces.push_back(faces[i]);	//inserted the unsorted one
		}
	}
	if (faces.size() != new_faces.size()) {
		faces = new_faces;
	}
	//printf("Removed %d-%d=%d duplicate faces of %d\n", (int)numFaces, (int)new_faces.size(), (int)numFaces-(int)new_faces.size(), (int)numFaces);

	return (unsigned int)new_faces.size();
}
unsigned int remove_degenerate_faces(std::vector<vec3i>& faces)
{
	std::vector<vec3i> new_faces;

	for (size_t i = 0; i < faces.size(); i++) {
		std::unordered_set<int> _set(3);
		bool foundDuplicate = false;
		if (_set.find(faces[i].x) != _set.end()) { foundDuplicate = true; } 
		else { _set.insert(faces[i].x); }                                 
		if (!foundDuplicate && _set.find(faces[i].y) != _set.end()) { foundDuplicate = true; } 
		else { _set.insert(faces[i].y); }                                 
		if (!foundDuplicate && _set.find(faces[i].z) != _set.end()) { foundDuplicate = true; } 
		else { _set.insert(faces[i].z); }
		if (!foundDuplicate) {
			new_faces.push_back(faces[i]);
		}
	}
	if (faces.size() != new_faces.size()) {
		faces = new_faces;
	}

	return (unsigned int)faces.size();
}
unsigned int hasNearestNeighbor( const vec3i& coord, SparseGrid3<std::list<std::pair<vec3f,unsigned int> > > &neighborQuery, const vec3f& v, float thresh )
{
	float threshSq = thresh*thresh;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i,j,k);
				if (neighborQuery.exists(c)) {
					for (const std::pair<vec3f, unsigned int>& n : neighborQuery[c]) {
						if (vec3f::distSq(v,n.first) < threshSq) {
							return n.second;
						}
					}
				}
			}
		}
	}
	return (unsigned int)-1;
}
unsigned int hasNearestNeighborApprox(const vec3i& coord, SparseGrid3<unsigned int> &neighborQuery) {
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			for (int k = -1; k <= 1; k++) {
				vec3i c = coord + vec3i(i,j,k);
				if (neighborQuery.exists(c)) {
					return neighborQuery[c];
				}
			}
		}
	}
	return (unsigned int)-1;
}
int sgn(float val) {
    return (0.0f < val) - (val < 0.0f);
}
std::pair<std::vector<vec3f>, std::vector<vec3i>> merge_close_vertices(const std::vector<Triangle>& meshTris, float thresh, bool approx)
{
	// assumes voxelsize = 1
	assert(thresh > 0);
	unsigned int numV = (unsigned int)meshTris.size() * 3;
	std::vector<vec3f> vertices(numV);
	std::vector<vec3i> faces(meshTris.size());
	for (int i = 0; i < (int)meshTris.size(); i++) {
		vertices[3*i+0].x = meshTris[i].v0.x;
		vertices[3*i+0].y = meshTris[i].v0.y;
		vertices[3*i+0].z = meshTris[i].v0.z;
		
		vertices[3*i+1].x = meshTris[i].v1.x;
		vertices[3*i+1].y = meshTris[i].v1.y;
		vertices[3*i+1].z = meshTris[i].v1.z;
		
		vertices[3*i+2].x = meshTris[i].v2.x;
		vertices[3*i+2].y = meshTris[i].v2.y;
		vertices[3*i+2].z = meshTris[i].v2.z;
		
		faces[i].x = 3*i+0;
		faces[i].y = 3*i+1;
		faces[i].z = 3*i+2;
	}

	std::vector<unsigned int> vertexLookUp;	vertexLookUp.resize(numV);
	std::vector<vec3f> new_verts; new_verts.reserve(numV);

	unsigned int cnt = 0;
	if (approx) {
		SparseGrid3<unsigned int> neighborQuery(0.6f, numV*2);
		for (unsigned int v = 0; v < numV; v++) {

			const vec3f& vert = vertices[v];
			vec3i coord = vec3i(vert.x/thresh + 0.5f*sgn(vert.x), vert.y/thresh + 0.5f*sgn(vert.y), vert.z/thresh + 0.5f*sgn(vert.z));			
			unsigned int nn = hasNearestNeighborApprox(coord, neighborQuery);

			if (nn == (unsigned int)-1) {
				neighborQuery[coord] = cnt;
				new_verts.push_back(vert);
				vertexLookUp[v] = cnt;
				cnt++;
			} else {
				vertexLookUp[v] = nn;
			}
		}
	} else {
		SparseGrid3<std::list<std::pair<vec3f, unsigned int> > > neighborQuery(0.6f, numV*2);
		for (unsigned int v = 0; v < numV; v++) {

			const vec3f& vert = vertices[v];
			vec3i coord = vec3i(vert.x/thresh + 0.5f*sgn(vert.x), vert.y/thresh + 0.5f*sgn(vert.y), vert.z/thresh + 0.5f*sgn(vert.z));
			unsigned int nn = hasNearestNeighbor(coord, neighborQuery, vert, thresh);

			if (nn == (unsigned int)-1) {
				neighborQuery[coord].push_back(std::make_pair(vert,cnt));
				new_verts.push_back(vert);
				vertexLookUp[v] = cnt;
				cnt++;
			} else {
				vertexLookUp[v] = nn;
			}
		}
	}
	// Update faces
	for (int i = 0; i < (int)faces.size(); i++) {		
		faces[i].x = vertexLookUp[faces[i].x];
		faces[i].y = vertexLookUp[faces[i].y];
		faces[i].z = vertexLookUp[faces[i].z];
	}

	if (vertices.size() != new_verts.size()) {
		vertices = new_verts;
	}

	remove_degenerate_faces(faces);
	//printf("Merged %d-%d=%d of %d vertices\n", numV, cnt, numV-cnt, numV);
	return std::make_pair(vertices, faces);
}
// ----- MESH CLEANUP FUNCTIONS

void run_marching_cubes_internal(
    const npy_accessor& tsdf_accessor,
	float isovalue,
	float truncation,
	float thresh,
	std::vector<Triangle>& results) {
	results.clear();

	for (int i = 0; i < (int)tsdf_accessor.size()[0]; i++) {
		for (int j = 0; j < (int)tsdf_accessor.size()[1]; j++) {
			for (int k = 0; k < (int)tsdf_accessor.size()[2]; k++) {
			extract_isosurface_at_position(vec3f(i, j, k), tsdf_accessor, truncation, isovalue, thresh, results);
			} // k
		} // j
	} // i
	//printf("#results = %d\n", (int)results.size());
}

void marching_cubes(const npy_accessor& tsdf_accessor, double isovalue, double truncation,
    std::vector<double>& vertices, std::vector<unsigned long>& polygons) {

    std::vector<Triangle> results;
    float thresh = 10.0f;
    run_marching_cubes_internal(tsdf_accessor, isovalue, truncation, thresh, results);

    // cleanup
	auto cleaned = merge_close_vertices(results, 0.00001f, true);
	remove_duplicate_faces(cleaned.second);

    vertices.resize(3 * cleaned.first.size());
    polygons.resize(3 * cleaned.second.size());

	for (int i = 0; i < (int)cleaned.first.size(); i++) {
	    vertices[3 * i + 0] = cleaned.first[i].x;
	    vertices[3 * i + 1] = cleaned.first[i].y;
	    vertices[3 * i + 2] = cleaned.first[i].z;
	}

	for (int i = 0; i < (int)cleaned.second.size(); i++) {
		polygons[3 * i + 0] = cleaned.second[i].x;
	    polygons[3 * i + 1] = cleaned.second[i].y;
	    polygons[3 * i + 2] = cleaned.second[i].z;
	}

}
