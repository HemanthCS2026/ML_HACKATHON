import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

def create_deck():
    prs = Presentation()
    # Set slide dimensions 16:9
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6] # blank

    # Color Palette
    PRIMARY = RGBColor(30, 58, 138)     # Navy #1E3A8A
    SECONDARY = RGBColor(5, 150, 105)   # Emerald #059669
    ACCENT = RGBColor(217, 119, 6)      # Amber #D97706
    DARK = RGBColor(31, 41, 55)         # Charcoal #1F2937
    LIGHT = RGBColor(243, 244, 246)     # Light Gray #F3F4F6
    WHITE = RGBColor(255, 255, 255)
    MUTED = RGBColor(107, 114, 128)     # Muted Gray

    def add_header(slide, title_text, category="PATHSNAP | PHASE 1 REVIEW"):
        # Category tag
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.4))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = SECONDARY

        # Title
        t_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
        tf_t = t_box.text_frame
        tf_t.word_wrap = True
        p_t = tf_t.paragraphs[0]
        p_t.text = title_text
        p_t.font.size = Pt(24)
        p_t.font.bold = True
        p_t.font.color.rgb = PRIMARY

        # Decorative bottom bar
        line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.5), Inches(11.733), Inches(0.04))
        line.fill.solid()
        line.fill.fore_color.rgb = RGBColor(229, 231, 235)
        line.line.color.rgb = RGBColor(229, 231, 235)

    def add_card(slide, left, top, width, height, bg_color=LIGHT, border_color=None):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        if border_color:
            shape.line.color.rgb = border_color
            shape.line.width = Pt(1.5)
        else:
            shape.line.fill.background()
        return shape

    # -------------------------------------------------------------
    # SLIDE 1: Title Slide
    # -------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    bg1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg1.fill.solid()
    bg1.fill.fore_color.rgb = PRIMARY
    bg1.line.fill.background()

    # Title box
    tbox = s1.shapes.add_textbox(Inches(1.0), Inches(1.0), Inches(11.333), Inches(2.2))
    tf1 = tbox.text_frame
    tf1.word_wrap = True
    
    p0 = tf1.paragraphs[0]
    p0.text = "UE24CS352A: MACHINE LEARNING HACKATHON | REVIEW 1"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(167, 243, 208) # Mint
    p0.space_after = Pt(10)

    p1 = tf1.add_paragraph()
    p1.text = "PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = WHITE
    p1.space_after = Pt(10)

    p2 = tf1.add_paragraph()
    p2.text = "End-to-End System Architecture, Empirical EDA & 5-Model Benchmark Evaluation"
    p2.font.size = Pt(18)
    p2.font.color.rgb = RGBColor(224, 231, 255)

    # Department info
    dept_box = s1.shapes.add_textbox(Inches(1.0), Inches(3.3), Inches(11.333), Inches(0.8))
    tf_dept = dept_box.text_frame
    p_dept = tf_dept.paragraphs[0]
    p_dept.text = "Department of Computer Science & Engineering | PES University (RR Campus)"
    p_dept.font.size = Pt(14)
    p_dept.font.color.rgb = RGBColor(209, 213, 219)

    # Team Members Card
    team_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.3), Inches(11.333), Inches(2.4))
    team_card.fill.solid()
    team_card.fill.fore_color.rgb = RGBColor(255, 255, 255)
    team_card.line.fill.background()

    tc_box = s1.shapes.add_textbox(Inches(1.2), Inches(4.4), Inches(10.9), Inches(2.1))
    tcf = tc_box.text_frame
    tcf.word_wrap = True

    p_th = tcf.paragraphs[0]
    p_th.text = "PROJECT TEAM MEMBERS"
    p_th.font.size = Pt(13)
    p_th.font.bold = True
    p_th.font.color.rgb = SECONDARY
    p_th.space_after = Pt(12)

    # Add 4 members in columns
    members = [
        ("HEMANTH V", "PES1UG25CS817"),
        ("PRAVEEN", "PES1UG25CS832"),
        ("AJAY T NAIK", "PES1UG24CS552"),
        ("AKULA SAINANDAN ROYAL", "PES1UG24CS553")
    ]
    for idx, (m_name, m_srn) in enumerate(members):
        col_left = Inches(1.3 + (idx * 2.7))
        m_box = s1.shapes.add_textbox(col_left, Inches(4.9), Inches(2.6), Inches(1.4))
        mtf = m_box.text_frame
        mtf.word_wrap = True
        mp1 = mtf.paragraphs[0]
        mp1.text = m_name
        mp1.font.size = Pt(15)
        mp1.font.bold = True
        mp1.font.color.rgb = PRIMARY
        
        mp2 = mtf.add_paragraph()
        mp2.text = m_srn
        mp2.font.size = Pt(13)
        mp2.font.color.rgb = DARK

    # -------------------------------------------------------------
    # SLIDE 2: Problem Statement & Real-World Challenges
    # -------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement & Real-World Challenges", "1. Problem Understanding")

    # 3 Cards
    cards_data = [
        ("GPS Signal Noise & Urban Canyons", [
            "Tall urban buildings cause severe multipath reflections, signal scattering, and satellite occlusion.",
            "Raw GPS observations jump across parallel streets, buildings, and highway ramps.",
            "Sampling interval is nominally 1 Hz with intermittent GPS dropout gaps."
        ], PRIMARY),
        ("Why Naive Matching Fails", [
            "Point-to-segment Euclidean distance alone frequently matches to wrong parallel streets.",
            "Opposite-direction one-way links are picked if vehicle direction is ignored.",
            "Produces physically impossible 'teleportation' jumps with disconnected road links."
        ], ACCENT),
        ("PathSnap Solution Architecture", [
            "Part 1 (Classification): High-speed spatial candidate generation + ML candidate classifier.",
            "Part 2 (Sequential Decoding): Hidden Markov Model (HMM) + Viterbi algorithm.",
            "Enforces road-network topological graph continuity for realistic trajectory recovery."
        ], SECONDARY)
    ]

    for idx, (head, bullets, col_theme) in enumerate(cards_data):
        c_left = Inches(0.8 + (idx * 3.95))
        c_top = Inches(1.8)
        card = add_card(s2, c_left, c_top, Inches(3.75), Inches(5.1), WHITE, col_theme)
        
        tbox = s2.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.2), Inches(3.35), Inches(4.7))
        tf = tbox.text_frame
        tf.word_wrap = True
        p_h = tf.paragraphs[0]
        p_h.text = head
        p_h.font.size = Pt(16)
        p_h.font.bold = True
        p_h.font.color.rgb = col_theme
        p_h.space_after = Pt(14)

        for b in bullets:
            p_b = tf.add_paragraph()
            p_b.text = f"• {b}"
            p_b.font.size = Pt(13)
            p_b.font.color.rgb = DARK
            p_b.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 3: Dataset Architecture & Exploratory Data Analysis
    # -------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Dataset Architecture & Empirical Data Exploration", "2. Dataset Understanding")

    # Left box: Textual insights
    left_card = add_card(s3, Inches(0.8), Inches(1.8), Inches(5.6), Inches(5.2), WHITE, PRIMARY)
    ltbox = s3.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.2), Inches(4.8))
    ltf = ltbox.text_frame
    ltf.word_wrap = True

    lp0 = ltf.paragraphs[0]
    lp0.text = "Dataset Structure & Verification"
    lp0.font.size = Pt(17)
    lp0.font.bold = True
    lp0.font.color.rgb = PRIMARY
    lp0.space_after = Pt(10)

    insights = [
        "Partition Scale: 80 Training Trips, 10 Student-Test Trips, 10 Evaluator-Test Trips.",
        ".track Files: Longitude, Latitude, Timestamp in seconds. Verified that 97.4% of intervals are exactly 1.0s (~1 Hz nominal rate).",
        ".nodes & .arcs Files: Represents localized road-network subgraphs per trip, ranging from 649 to 307,392 directed links.",
        ".route Files: Chronological sequence of ground-truth arc IDs. Verified 100% graph continuity on reference routes.",
        "Velocity Profile: Trajectory median speed is 13.8 m/s (49.7 km/h) with highway segments exceeding 25 m/s."
    ]
    for ins in insights:
        p = ltf.add_paragraph()
        p.text = f"✔ {ins}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK
        p.space_after = Pt(8)

    # Right side: Embed EDA image
    eda_img = "outputs/eda_analysis_plots.png"
    if os.path.exists(eda_img):
        s3.shapes.add_picture(eda_img, Inches(6.6), Inches(1.8), Inches(5.9), Inches(5.2))

    # -------------------------------------------------------------
    # SLIDE 4: Spatial Candidate Generation & Preprocessing
    # -------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Data Preprocessing & Spatial Candidate Generation", "3. Preprocessing Pipeline")

    steps = [
        ("Step 1: Geodesic Metric Projection", [
            "Latitude & Longitude are angular coordinates; Euclidean calculations in degrees are invalid.",
            "We project coordinates into local metric Cartesian coordinates (x, y) in meters centered at trajectory median:",
            "x = (lon - lon_center) * 111139 * cos(lat_center)",
            "y = (lat - lat_center) * 111139"
        ]),
        ("Step 2: KD-Tree Spatial Indexing", [
            "Road network graph consists of tens of thousands of directed segments.",
            "Direct distance scanning over all links is computationally prohibitive (O(N*M)).",
            "We construct a scipy.spatial.KDTree over all road vertices to query K=20 nearest nodes in O(log M) time.",
            "Gather incident directed arcs to form the localized candidate set."
        ]),
        ("Step 3: Exact Perpendicular Projection & Recall", [
            "Compute orthogonal point-to-segment distance and clamp projection to [0, 1].",
            "Sort candidate segments by physical distance and retain Top-5 candidates per GPS point.",
            "Candidate Miss Rate: Only 3.49% (Verified Top-5 Candidate Recall of 96.51%)."
        ])
    ]

    for idx, (st_title, st_bullets) in enumerate(steps):
        c_left = Inches(0.8 + (idx * 3.95))
        c_top = Inches(1.8)
        add_card(s4, c_left, c_top, Inches(3.75), Inches(5.1), WHITE, SECONDARY)
        tbox = s4.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.2), Inches(3.35), Inches(4.7))
        tf = tbox.text_frame
        tf.word_wrap = True

        p_h = tf.paragraphs[0]
        p_h.text = st_title
        p_h.font.size = Pt(15)
        p_h.font.bold = True
        p_h.font.color.rgb = SECONDARY
        p_h.space_after = Pt(12)

        for b in st_bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(12)
            p.font.color.rgb = DARK
            p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 5: Feature Engineering Pipeline
    # -------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Feature Engineering: Physical & Behavioral Justifications", "4. Feature Engineering")

    feats = [
        ("dist_m", "Distance to Segment (m)", "Euclidean perpendicular distance from GPS observation to candidate segment."),
        ("proj_ratio", "Projection Ratio [0, 1]", "Location along segment where observation projects; penalizes overshooting endpoints."),
        ("heading_diff_deg", "Heading Difference (°)", "Angular deviation between vehicle travel heading and road arc orientation (0° to 180°)."),
        ("heading_alignment", "Cosine Alignment [-1, 1]", "cos(Δθ); smoothly rewards forward alignment (+1) and heavily penalizes wrong-way links (-1)."),
        ("arc_len", "Arc Physical Length (m)", "Segment length; normalizes capture basins between short intersection links and long highway arcs."),
        ("speed_mps", "Instantaneous Velocity (m/s)", "Vehicle velocity Δs/Δt; identifies stopped vehicles, traffic congestion, and highway cruise."),
        ("spatial_prior", "Gaussian Emission Prior", "exp(-d² / 2σ²) with σ=6.0m; standard physical GPS sensor noise model (Newson & Krumm)."),
        ("distance_rank", "Candidate Rank (0 to 4)", "Ordinal proximity priority among the 5 retrieved candidates.")
    ]

    for idx, (fname, flabel, fdesc) in enumerate(feats):
        row = idx // 2
        col = idx % 2
        c_left = Inches(0.8 + (col * 5.95))
        c_top = Inches(1.8 + (row * 1.3))
        
        card = add_card(s5, c_left, c_top, Inches(5.75), Inches(1.15), WHITE, PRIMARY)
        tbox = s5.shapes.add_textbox(c_left + Inches(0.15), c_top + Inches(0.1), Inches(5.45), Inches(0.95))
        tf = tbox.text_frame
        tf.word_wrap = True

        p1 = tf.paragraphs[0]
        p1.text = f"{idx+1}. {flabel} (`{fname}`)"
        p1.font.size = Pt(13)
        p1.font.bold = True
        p1.font.color.rgb = PRIMARY
        p1.space_after = Pt(3)

        p2 = tf.add_paragraph()
        p2.text = fdesc
        p2.font.size = Pt(11)
        p2.font.color.rgb = DARK

    # -------------------------------------------------------------
    # SLIDE 6: 5-Model Benchmark Table (Official Results)
    # -------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "5-Model Benchmark Leaderboard & Comparison", "5. Model Experimentation")

    # Add Table
    rows = 6
    cols = 7
    tbl_shape = s6.shapes.add_table(rows, cols, Inches(0.8), Inches(1.8), Inches(11.733), Inches(3.2))
    tbl = tbl_shape.table

    # Column widths
    tbl.columns[0].width = Inches(2.2)
    tbl.columns[1].width = Inches(1.9)
    tbl.columns[2].width = Inches(1.6)
    tbl.columns[3].width = Inches(1.6)
    tbl.columns[4].width = Inches(1.5)
    tbl.columns[5].width = Inches(1.5)
    tbl.columns[6].width = Inches(1.4)

    headers = ["Model", "Algorithm Family", "CV Macro F1", "Test Top-1 Acc", "Test Macro F1", "Log Loss", "Route Cont."]
    for c_idx, h in enumerate(headers):
        cell = tbl.cell(0, c_idx)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER

    data = [
        ["Logistic Regression", "Linear & Probabilistic", "0.7975", "83.97%", "0.8185", "0.2987", "88.7%"],
        ["Decision Tree", "Tree-Based", "0.8237", "84.08%", "0.8406", "0.2502", "90.1%"],
        ["Random Forest", "Ensemble (Bagging)", "0.8361", "84.43%*", "0.8483", "0.2249", "90.6%*"],
        ["LightGBM", "Gradient Boosting", "0.8391", "83.85%", "0.8447", "0.2276", "90.6%*"],
        ["Multi-Layer Perceptron", "Neural Networks", "0.8489*", "83.52%", "0.8866*", "0.1744*", "89.4%"]
    ]

    for r_idx, row_vals in enumerate(data):
        for c_idx, val in enumerate(row_vals):
            cell = tbl.cell(r_idx + 1, c_idx)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(249, 250, 251) if r_idx % 2 == 0 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.font.color.rgb = DARK
                p.alignment = PP_ALIGN.CENTER
                if "*" in val:
                    p.font.bold = True
                    p.font.color.rgb = SECONDARY

    # Bottom summary card
    b_card = add_card(s6, Inches(0.8), Inches(5.3), Inches(11.733), Inches(1.7), WHITE, SECONDARY)
    btbox = s6.shapes.add_textbox(Inches(1.0), Inches(5.4), Inches(11.3), Inches(1.5))
    btf = btbox.text_frame
    btf.word_wrap = True

    bp1 = btf.paragraphs[0]
    bp1.text = "Key Findings & Observations:"
    bp1.font.size = Pt(13)
    bp1.font.bold = True
    bp1.font.color.rgb = SECONDARY
    bp1.space_after = Pt(4)

    finds = [
        "Top Observation Accuracy: Random Forest achieves the highest student-test Top-1 accuracy at 84.43%, closely followed by Decision Tree (84.08%) and Logistic Regression (83.97%).",
        "Probability Calibration & Loss: MLP achieves the lowest Log Loss (0.1744) and highest Macro F1 (0.8866), showing superior posterior probability confidence estimates.",
        "Route Continuity: Ensemble models (Random Forest & LightGBM) lead route continuity at 90.6%, successfully preserving link connectivity without sequential decoding."
    ]
    for f in finds:
        p = btf.add_paragraph()
        p.text = f"• {f}"
        p.font.size = Pt(11)
        p.font.color.rgb = DARK

    # -------------------------------------------------------------
    # SLIDE 7: Benchmark Visualizations (Plots)
    # -------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    add_header(s7, "Model Comparison Diagnostic Visualizations", "6. Experimental Visuals")

    comp_img = "outputs/model_comparison_plots.png"
    if os.path.exists(comp_img):
        s7.shapes.add_picture(comp_img, Inches(0.8), Inches(1.8), Inches(8.0), Inches(5.2))

    # Right side explanation
    r_card = add_card(s7, Inches(9.0), Inches(1.8), Inches(3.533), Inches(5.2), WHITE, PRIMARY)
    rtbox = s7.shapes.add_textbox(Inches(9.2), Inches(2.0), Inches(3.133), Inches(4.8))
    rtf = rtbox.text_frame
    rtf.word_wrap = True

    rp0 = rtf.paragraphs[0]
    rp0.text = "Generalization & Stability"
    rp0.font.size = Pt(16)
    rp0.font.bold = True
    rp0.font.color.rgb = PRIMARY
    rp0.space_after = Pt(12)

    diag_pts = [
        "CV vs Test Alignment: Cross-validation Macro F1 tightly tracks student-test performance across all 5 models.",
        "Zero Catastrophic Overfitting: Test Top-1 Accuracy remains within ~2% of CV Top-1 scores.",
        "Tree Regularization: Controlled via max_depth=12 and min_samples_leaf=20 to prevent memorizing specific road geometries.",
        "Inference Speed: All models achieve < 0.06 ms per observation, well suited for real-time edge vehicle tracking."
    ]
    for d in diag_pts:
        p = rtf.add_paragraph()
        p.text = f"✔ {d}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK
        p.space_after = Pt(10)

    # -------------------------------------------------------------
    # SLIDE 8: Viva Questions & Technical Defense
    # -------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    add_header(s8, "Compulsory Viva Defense & Questions", "7. Viva Preparation")

    viva_cards = [
        ("Q1: Bias-Variance & Model Fit", [
            "Logistic Regression: Suffers from slight underfitting (CV Top-1 83.15%) because linear hyperplanes cannot isolate multi-dimensional heading-distance cones.",
            "Decision Tree & Random Forest: Effectively capture non-linear feature splits; bagging prevents overfitting to anomalous GPS pings.",
            "MLP: Benefits from early stopping on validation loss, achieving highest Macro F1 (0.8866)."
        ]),
        ("Q2: Why Observation Top-1 != Route Continuity", [
            "Observation Top-1 Accuracy measures independent point-level correctness in isolation.",
            "Two consecutive GPS points may be matched to disconnected road segments that lack network topological adjacency.",
            "Baseline classification achieves 90.6% continuity; the remaining ~9.4% gap requires sequential HMM Viterbi decoding."
        ]),
        ("Q3: Model Selection for Phase 2", [
            "We select Random Forest & LightGBM as primary candidate models for Bayesian Optimization.",
            "Reasons: Highest top-1 accuracy (84.43%), superior route continuity (90.6%), and rapid inference speed (0.04 ms/obs)."
        ])
    ]

    for idx, (vq_title, vq_bullets) in enumerate(viva_cards):
        c_left = Inches(0.8 + (idx * 3.95))
        c_top = Inches(1.8)
        add_card(s8, c_left, c_top, Inches(3.75), Inches(5.1), WHITE, PRIMARY)
        tbox = s8.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.2), Inches(3.35), Inches(4.7))
        tf = tbox.text_frame
        tf.word_wrap = True

        p_h = tf.paragraphs[0]
        p_h.text = vq_title
        p_h.font.size = Pt(15)
        p_h.font.bold = True
        p_h.font.color.rgb = PRIMARY
        p_h.space_after = Pt(12)

        for b in vq_bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(12)
            p.font.color.rgb = DARK
            p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 9: System Demo & Evaluator UI
    # -------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    add_header(s9, "Interactive Evaluator UI & Live Map Visualizer", "8. Live System Demo")

    # Left card
    ui_card = add_card(s9, Inches(0.8), Inches(1.8), Inches(5.75), Inches(5.2), WHITE, SECONDARY)
    uitbox = s9.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.35), Inches(4.8))
    uitf = uitbox.text_frame
    uitf.word_wrap = True

    up0 = uitf.paragraphs[0]
    up0.text = "Interactive Streamlit Dashboard (`app.py`)"
    up0.font.size = Pt(16)
    up0.font.bold = True
    up0.font.color.rgb = SECONDARY
    up0.space_after = Pt(12)

    ui_feats = [
        "Live Trip Selector: Evaluators can select any of the 10 Student-Test trips (e.g., 00000000, 00000007, 00000040) or 80 training trips.",
        "Model Switcher: Select between Random Forest, LightGBM, MLP, Decision Tree, or Logistic Regression.",
        "Instant Inference: Evaluates candidate generation, feature scaling, and model prediction in real time.",
        "Multi-Layer Interactive Folium Map: Visualizes original GPS pings (blue), road network graph (gray), reference ground truth (green), and predicted route (orange dashed).",
        "Leaderboard Tab: Directly displays the official 5-model benchmark metrics and comparative plots."
    ]
    for u in ui_feats:
        p = uitf.add_paragraph()
        p.text = f"✔ {u}"
        p.font.size = Pt(12)
        p.font.color.rgb = DARK
        p.space_after = Pt(8)

    # Right Card: Demo Instructions
    demo_card = add_card(s9, Inches(6.75), Inches(1.8), Inches(5.783), Inches(5.2), PRIMARY)
    dtbox = s9.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.283), Inches(4.8))
    dtf = dtbox.text_frame
    dtf.word_wrap = True

    dp0 = dtf.paragraphs[0]
    dp0.text = "Live Execution & Access Links"
    dp0.font.size = Pt(16)
    dp0.font.bold = True
    dp0.font.color.rgb = WHITE
    dp0.space_after = Pt(14)

    demo_steps = [
        ("Local Dashboard URL", "http://localhost:8501"),
        ("Network URL (Laptop 2 / Evaluator)", "http://10.20.203.229:8501"),
        ("Interactive Folium Maps", "maps/trip_train_00000001.html\nmaps/trip_test_00000000.html"),
        ("Official GitHub Repository", "https://github.com/HemanthCS2026/ML_HACKATHON")
    ]
    for d_title, d_val in demo_steps:
        p1 = dtf.add_paragraph()
        p1.text = d_title.upper()
        p1.font.size = Pt(11)
        p1.font.bold = True
        p1.font.color.rgb = RGBColor(167, 243, 208)
        p1.space_after = Pt(2)

        p2 = dtf.add_paragraph()
        p2.text = d_val
        p2.font.size = Pt(13)
        p2.font.color.rgb = WHITE
        p2.space_after = Pt(12)

    # -------------------------------------------------------------
    # SLIDE 10: Phase 2 Roadmap (After Lunch Break)
    # -------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    add_header(s10, "Roadmap for Phase 2 (After Lunch Break)", "9. Next Milestones")

    p2_steps = [
        ("1. Bayesian Optimization (Optuna)", [
            "Tune max_depth, num_leaves, n_estimators, and learning_rate for Random Forest & LightGBM.",
            "Record optimization search logs across group-aware cross-validation folds.",
            "Compare tuned vs. untuned performance."
        ]),
        ("2. HMM & Viterbi Decoding", [
            "Formulate road segments as hidden states and observations as GPS pings.",
            "Emission Probabilities: Derived from calibrated classifier predictions & Gaussian spatial priors.",
            "Transition Probabilities: Network shortest-path distance vs. Euclidean great-circle distance.",
            "Viterbi algorithm decodes globally optimal sequence, eliminating disjoint jumps (100% continuity)."
        ]),
        ("3. Distributed 2-Laptop Deployment", [
            "Laptop 1: FastAPI server hosting trained models and server-side ground truth.",
            "Laptop 2: Evaluator UI communicating with Laptop 1 via REST API.",
            "Complete final technical report (PDF) with verification."
        ])
    ]

    for idx, (p_title, p_bullets) in enumerate(p2_steps):
        c_left = Inches(0.8 + (idx * 3.95))
        c_top = Inches(1.8)
        add_card(s10, c_left, c_top, Inches(3.75), Inches(5.1), WHITE, SECONDARY)
        tbox = s10.shapes.add_textbox(c_left + Inches(0.2), c_top + Inches(0.2), Inches(3.35), Inches(4.7))
        tf = tbox.text_frame
        tf.word_wrap = True

        p_h = tf.paragraphs[0]
        p_h.text = p_title
        p_h.font.size = Pt(15)
        p_h.font.bold = True
        p_h.font.color.rgb = SECONDARY
        p_h.space_after = Pt(12)

        for b in p_bullets:
            p = tf.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(12)
            p.font.color.rgb = DARK
            p.space_after = Pt(8)

    # -------------------------------------------------------------
    # SLIDE 11: Conclusion & Q&A
    # -------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    bg11 = s11.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg11.fill.solid()
    bg11.fill.fore_color.rgb = PRIMARY
    bg11.line.fill.background()

    cbox = s11.shapes.add_textbox(Inches(1.5), Inches(1.8), Inches(10.333), Inches(4.0))
    ctf = cbox.text_frame
    ctf.word_wrap = True

    cp0 = ctf.paragraphs[0]
    cp0.text = "PATHSNAP | PHASE 1 REVIEW"
    cp0.font.size = Pt(16)
    cp0.font.bold = True
    cp0.font.color.rgb = RGBColor(167, 243, 208)
    cp0.alignment = PP_ALIGN.CENTER
    cp0.space_after = Pt(14)

    cp1 = ctf.add_paragraph()
    cp1.text = "Thank You!"
    cp1.font.size = Pt(44)
    cp1.font.bold = True
    cp1.font.color.rgb = WHITE
    cp1.alignment = PP_ALIGN.CENTER
    cp1.space_after = Pt(14)

    cp2 = ctf.add_paragraph()
    cp2.text = "Phase 1 Complete: Full EDA, Spatial Indexing, 5-Model Benchmark & Live Evaluator UI"
    cp2.font.size = Pt(18)
    cp2.font.color.rgb = RGBColor(224, 231, 255)
    cp2.alignment = PP_ALIGN.CENTER
    cp2.space_after = Pt(24)

    cp3 = ctf.add_paragraph()
    cp3.text = "Open for Questions & Live Demonstration"
    cp3.font.size = Pt(20)
    cp3.font.bold = True
    cp3.font.color.rgb = RGBColor(253, 224, 71) # Yellow
    cp3.alignment = PP_ALIGN.CENTER

    out_file = "PathSnap_Review_1_Presentation.pptx"
    prs.save(out_file)
    print(f"Successfully generated PowerPoint presentation: {out_file}")

if __name__ == "__main__":
    create_deck()
