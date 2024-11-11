import streamlit.web.cli as stcli
import sys
import os

if __name__ == "__main__":
    app_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "flight_analysis",
        "streamlit",
        "app.py"
    )
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())