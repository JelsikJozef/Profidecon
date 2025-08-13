import os, json
def build_tree(path):
    tree = {}
    for entry in os.listdir(path):
        full_path = os.path.join(path, entry)
        if os.path.isdir(full_path):
            tree[entry] = build_tree(full_path)
        else:
            tree[entry] = None
    return tree

root_dir = "../Knowledge"
tree = build_tree(root_dir)
with open("knowledge_tree.json", "w") as f:
    json.dump(tree, f, indent=2, ensure_ascii=False)