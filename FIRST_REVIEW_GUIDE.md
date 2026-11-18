# PathSnap: Evaluator Demo Guide & Viva Cheat-Sheet

**Course:** UE24CS352A Machine Learning  
**Department:** Computer Science & Engineering, PES University (RR Campus)  
**Topic:** PathSnap: Distributed Map-Matching & Route Reconstruction  

---

## 1. How to Run the Demo for Evaluators

### Setup 1: Two-Laptop Mode (Official)
1. Both laptops on the same Wi-Fi / hotspot.
2. **On Laptop 1 (Server):**
   ```bash
   python api.py
   ```
   *FastAPI starts on port 8000. Keep this terminal visible — evaluators will look for live incoming request logs.*
3. **On Laptop 2 (Client UI):**
   ```bash
   streamlit run app.py
   ```
   *In the sidebar, enter Laptop 1's IP: `http://<laptop-1-ip>:8000`.*
   *Pick any of the 10 student-test trips (e.g. `00000000`, `00000040`), select a model, make sure HMM is enabled, and click **"Run Map-Matching & Route Reconstruction"**.*
   *Show the live Top-1 accuracy (~88-96%), continuity (~97-99%), and the interactive Folium map.*

### Setup 2: Single Laptop (Fallback)
If Wi-Fi is spotty:
```bash
streamlit run app.py
```
Switch to "Local Standalone" in the sidebar. Everything runs locally from the saved artifacts in `artifacts/`.

---

## 2. Key Numbers to Remember

- **Dataset:** 80 training trips, 10 student-test trips, 10 reserved evaluator trips.
- **Sampling Rate:** ~1 Hz (97% of GPS intervals are exactly 1 second).
- **Candidate Retrieval:** KDTree on road nodes. Query 20 nearest nodes, extract incident arcs, pick top 5.
- **Candidate Miss Rate:** **3.49%** (Top-5 candidate recall is **96.51%**).
- **Best Classifier:** **Random Forest** (84.43% test Top-1, 90.6% continuity).
- **Fastest Model:** **LightGBM** (0.039 ms/obs).
- **Best Log Loss / Calibrated Probs:** **MLP** (0.1744 log loss, 0.8866 Macro F1).
- **Bayesian Tuning:** 25 Optuna TPE trials on LightGBM. Improved CV Macro F1 from 0.8391 to **0.8590**.
- **HMM Impact:** Boosts continuity from **90.62% to 97.23%** (+6.6%) and Top-1 accuracy from **86.54% to 89.66%** (+3.1%).

---

## 3. Likely Viva Questions & Clean Answers

### Q: Why do you need coordinate projection? Why not just use (lat, lon) directly?
> "Latitude and longitude are spherical angles, not flat coordinates. 1 degree of longitude shrinks as you move north or south (`dx = dlon * 111139 * cos(lat)`). If you take Euclidean distance directly on degrees, east-west distances get distorted compared to north-south distances. We project into local Cartesian meters centered at the trip's median position so all distances, headings, and speeds are physically accurate."

### Q: Why not test all road segments in the map for every GPS point?
> "Some road networks have over 300,000 links. Checking all links for 2,500 GPS points would mean over 750 million distance calculations per trip. Instead, we build a KDTree on the road nodes, find the 20 nearest nodes in O(log N) time, and only test their incident arcs. This drops computation to milliseconds while keeping a 96.5% candidate recall."

### Q: What features did you engineer and why?
> "We engineered 8 features:
> 1. `dist_m`: Distance from GPS point to the road segment.
> 2. `proj_ratio`: Where along the segment the point projects (0 to 1).
> 3. `heading_diff_deg` & `heading_align`: Vehicle movement heading vs road heading. `cos(diff)` gives +1 for forward travel and -1 for wrong-way driving.
> 4. `arc_len`: Length of the road segment, normalizing capture zones.
> 5. `speed_mps`: Instantaneous vehicle speed.
> 6. `spatial_prior`: Gaussian sensor noise density (`sigma = 6m`).
> 7. `distance_rank`: Rank among the 5 local candidates (0 to 4)."

### Q: Why did you pick Random Forest / LightGBM over Logistic Regression?
> "Map matching decisions involve threshold combinations — like *'distance < 8m AND heading alignment > 0.7'*. Decision trees and tree ensembles create rectangular decision boundaries that fit these thresholds naturally. Logistic regression tries to separate classes with a single linear boundary, which slightly underfits because heading and distance interact non-linearly."

### Q: Why is 84% observation accuracy not enough? Why do you need an HMM?
> "Observation-level accuracy looks at each GPS point independently. Even with 84% accuracy, about 10% of the time consecutive points get assigned to road segments that don't connect in the road network graph (like jumping from a flyover to the road below). An HMM looks at the whole sequence: it uses the classifier's score as the emission probability and the road graph connectivity as the transition probability, finding a path that is both close to the GPS pings and continuous on the road network."

### Q: How do you handle large GPS gaps in the HMM?
> "If timestamps show a gap greater than 10 seconds (a dropout), the car could have taken multiple turns while offline. We relax the transition penalty so the Viterbi decoder doesn't force an impossible route connection across the gap."

### Q: How does your two-laptop deployment prevent data leakage?
> "Laptop 2 (the evaluator UI) only sends the selected trip ID to Laptop 1 over HTTP. Laptop 1 does the candidate extraction, model inference, and HMM decoding, and compares against the server-side ground truth route. The ground-truth route file never gets downloaded or exposed to Laptop 2."
