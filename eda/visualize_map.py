import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import folium
from folium import plugins
import numpy as np

def create_folium_map(trip_data, predicted_route=None, output_html="maps/trip_map.html"):
    """
    Generates an interactive Folium map displaying:
    1. Road network graph arcs (gray)
    2. Ground-truth reference route (green)
    3. Raw GPS observations (blue points with tooltips)
    4. Predicted route (orange/red dashed polyline, if provided)
    """
    track = trip_data["track"]      # [lon, lat, ts]
    nodes = trip_data["nodes"]      # [lon, lat]
    arcs = trip_data["arcs"]        # [u, v]
    route = trip_data["route"]      # [arc_id, ...]
    trip_id = trip_data["trip_id"]

    # Compute bounding box and center
    center_lat = float(np.median(track[:, 1]))
    center_lon = float(np.median(track[:, 0]))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=15,
        tiles="OpenStreetMap"
    )

    # Layer 1: Road Network Arcs (subsampled or bounded near trajectory to keep file responsive)
    min_lon, max_lon = track[:, 0].min() - 0.015, track[:, 0].max() + 0.015
    min_lat, max_lat = track[:, 1].min() - 0.015, track[:, 1].max() + 0.015

    network_group = folium.FeatureGroup(name="Road Network Arcs (Nearby)", show=True)
    
    # Filter arcs that fall within the trajectory bounding box
    u_nodes = arcs[:, 0]
    v_nodes = arcs[:, 1]
    u_coords = nodes[u_nodes]
    v_coords = nodes[v_nodes]

    in_box = (
        (u_coords[:, 0] >= min_lon) & (u_coords[:, 0] <= max_lon) &
        (u_coords[:, 1] >= min_lat) & (u_coords[:, 1] <= max_lat) &
        (v_coords[:, 0] >= min_lon) & (v_coords[:, 0] <= max_lon) &
        (v_coords[:, 1] >= min_lat) & (v_coords[:, 1] <= max_lat)
    )

    # Subsample if too dense to avoid multi-MB html
    selected_arc_indices = np.where(in_box)[0]
    if len(selected_arc_indices) > 3000:
        selected_arc_indices = np.random.choice(selected_arc_indices, 3000, replace=False)

    for arc_idx in selected_arc_indices:
        u = arcs[arc_idx, 0]
        v = arcs[arc_idx, 1]
        lat1, lon1 = float(nodes[u, 1]), float(nodes[u, 0])
        lat2, lon2 = float(nodes[v, 1]), float(nodes[v, 0])
        folium.PolyLine(
            locations=[[lat1, lon1], [lat2, lon2]],
            color="#888888",
            weight=1.5,
            opacity=0.4
        ).add_to(network_group)

    network_group.add_to(m)

    # Layer 2: Ground Truth Route
    if route is not None and len(route) > 0:
        route_group = folium.FeatureGroup(name="Ground Truth Route (Green)", show=True)
        route_lines = []
        for arc_id in route:
            u, v = arcs[arc_id]
            lat1, lon1 = float(nodes[u, 1]), float(nodes[u, 0])
            lat2, lon2 = float(nodes[v, 1]), float(nodes[v, 0])
            route_lines.append([[lat1, lon1], [lat2, lon2]])

        folium.PolyLine(
            locations=route_lines,
            color="#10B981",
            weight=5,
            opacity=0.85,
            tooltip="Ground-Truth Traversed Route"
        ).add_to(route_group)
        route_group.add_to(m)

    # Layer 3: Raw GPS Trajectory (Observations)
    gps_group = folium.FeatureGroup(name="Raw GPS Observations (Blue)", show=True)
    gps_pts = [[float(pt[1]), float(pt[0])] for pt in track]
    folium.PolyLine(
        locations=gps_pts,
        color="#3B82F6",
        weight=2,
        opacity=0.6,
        dash_array="4, 4"
    ).add_to(gps_group)

    # Add circle markers for individual GPS pings (subsampled for rendering speed)
    step = max(1, len(track) // 150)
    for idx in range(0, len(track), step):
        lat = float(track[idx, 1])
        lon = float(track[idx, 0])
        ts = float(track[idx, 2])
        folium.CircleMarker(
            location=[lat, lon],
            radius=3.5,
            color="#1E40AF",
            fill=True,
            fill_color="#60A5FA",
            fill_opacity=0.9,
            tooltip=f"GPS Ping #{idx} (t={ts}s)"
        ).add_to(gps_group)

    # Mark Start and End Points
    folium.Marker(
        location=[float(track[0, 1]), float(track[0, 0])],
        popup="Start Point",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(gps_group)

    folium.Marker(
        location=[float(track[-1, 1]), float(track[-1, 0])],
        popup="End Point",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(gps_group)

    gps_group.add_to(m)

    # Layer 4: Predicted Route (if provided)
    if predicted_route is not None and len(predicted_route) > 0:
        pred_group = folium.FeatureGroup(name="Predicted Route (Orange)", show=True)
        pred_lines = []
        for arc_id in predicted_route:
            u, v = arcs[arc_id]
            lat1, lon1 = float(nodes[u, 1]), float(nodes[u, 0])
            lat2, lon2 = float(nodes[v, 1]), float(nodes[v, 0])
            pred_lines.append([[lat1, lon1], [lat2, lon2]])

        folium.PolyLine(
            locations=pred_lines,
            color="#F97316",
            weight=4,
            opacity=0.9,
            dash_array="6, 6",
            tooltip="Predicted Trajectory"
        ).add_to(pred_group)
        pred_group.add_to(m)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)

    # Save HTML
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    m.save(output_html)
    print(f"Saved interactive Folium map to: {output_html}")
    return output_html

if __name__ == "__main__":
    import sys
    from src.data_loader import load_trip
    trip_id = sys.argv[1] if len(sys.argv) > 1 else "00000001"
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    trip = load_trip(trip_id, split=split)
    out_path = f"maps/trip_{trip_id}_interactive.html"
    create_folium_map(trip, output_html=out_path)
