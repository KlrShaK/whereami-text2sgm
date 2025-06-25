"""
Take Reference from visualization_graph-object.py to know how to find match scene and text graph and also find common objects between them.

Algorithm:
1. We obtain the scene graph and the text graph.
2. Calculate cosine similarity between all nodes in the scene and text graphs.
3. Threshold the similarities to obtain top 5 potential matches.
4. Load the 3D meshes of the scene.
5. For Loop through the top 5 matches:
    5.1 For each match, find the object instances from text-graph in the 3D mesh.
    5.2 intantiate 36 possible particles/positions uniformly (360 FoV camera) in the 3D mesh scene.
    5.3 At each possible postion, check which particles can see, each object instance (using ray-casting)
    5.4 Give each possible particle a score based on how many objects it can see.
    5.5 Create a probability heatmatrix of the particles based on the scores.

"""