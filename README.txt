==============================================
  BRAIN STROKE PREDICTION PROJECT
==============================================

FOLDER CONTENTS:
----------------
  app.py                               ← Main Streamlit app
  stroke_cardiovascular_synthetic.csv  ← CSV data (already inside)
  brain_model.h5                       ← ⚠️ YOU MUST ADD THIS (see below)
  RUN_APP.bat                          ← Double-click this to launch!

----------------------------------------------
STEP 1 — Install requirements (only once)
----------------------------------------------
Open CMD and run:

  pip install streamlit pandas numpy scikit-learn pillow tensorflow

----------------------------------------------
STEP 2 — Save your brain model
----------------------------------------------
In your Jupyter notebook, after training, run:

  model.save("brain_model.h5")

Then COPY that brain_model.h5 file INTO this folder.

----------------------------------------------
STEP 3 — Run the app
----------------------------------------------
Just DOUBLE-CLICK  →  RUN_APP.bat

Your browser will open automatically at:
  http://localhost:8501

----------------------------------------------
PAGES:
----------------------------------------------
🩺 ML Prediction      → Fill patient details → Predict stroke risk
🧠 Brain Image        → Upload brain scan   → Normal or Stroke
==============================================
