from flask import Flask, request, jsonify
from math import radians, sin, cos, sqrt, atan2

app = Flask(__name__)

# Function to calculate distance between two GPS points
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c  # Distance in km

@app.route('/verify_walk', methods=['POST'])
def verify_walk():
    data = request.get_json()
    coords = data.get("coordinates", [])  # list of {lat, lon}

    if len(coords) < 2:
        return jsonify({"error": "Not enough coordinates"}), 400

    total_distance = 0.0
    for i in range(len(coords) - 1):
        lat1, lon1 = coords[i]["lat"], coords[i]["lon"]
        lat2, lon2 = coords[i+1]["lat"], coords[i+1]["lon"]
        total_distance += haversine(lat1, lon1, lat2, lon2)

    threshold = 2.0  # km
    result = total_distance >= threshold

    return jsonify({
        "total_distance_km": round(total_distance, 2),
        "walk_valid": result
    })

if __name__ == '__main__':
    app.run(debug=True)
