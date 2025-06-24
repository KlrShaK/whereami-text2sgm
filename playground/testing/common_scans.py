import json

# Paths to the two JSON files that were uploaded
path_3r = './3RScan.json'
path_data = './data_json_format.json'

# ------------------------------------------------------------------
# 1. Read 3RScan.json and build the list‑of‑lists with scan IDs that
#    come from the SAME physical place (reference + its re‑scans)
# ------------------------------------------------------------------
with open(path_3r, 'r') as f:
    scenes = json.load(f)

same_place_groups = []
for scene in scenes:
    # start with the reference scan
    ids = [scene['reference']]
    # add every rescan's reference id
    ids.extend(scan_entry['reference'] for scan_entry in scene['scans'])
    same_place_groups.append(ids)

# Show the first 5 groups, just for a quick visual check
print("First 5 scan‑ID groups (reference + rescans)--i.e scans belonging to the same place-- :")
for g in same_place_groups[:5]:
    print(" ", g)

# ------------------------------------------------------------------
# 2. Filter those groups by what appears in data_json_format.json
#    Keep only groups where AT LEAST TWO scan IDs occur in that file
# ------------------------------------------------------------------
with open(path_data, 'r') as f:
    description_entries = json.load(f)

scan_ids_in_data = {entry['scanId'] for entry in description_entries}

# filtered_groups = [
#     g for g in same_place_groups
#     if sum(scan_id in scan_ids_in_data for scan_id in g) >= 2
# ]

filtered_groups = [
    [scan_id for scan_id in g if scan_id in scan_ids_in_data]
    for g in same_place_groups
    if sum(scan_id in scan_ids_in_data for scan_id in g) >= 2
]


print("\nNumber of groups with ≥2 matches in data_json_format.json:", len(filtered_groups))
# print("First 5 of those groups (with the matching IDs highlighted):")
for g in filtered_groups:
    present = [s for s in g if s in scan_ids_in_data]
    # print(" ", g, "\n --> present in data_json_format.json:", present)
    print(" ", "\n --> present in scanscribe:", g)
