import io
import base64
import traceback
import pandas as pd
import matplotlib.pyplot as plt

def run_generated_code(code_str):
    try:
        allowed_builtins = {"__builtins__": {"print": print, "len": len, "range": range}}
        local_vars = {}
        exec(code_str, allowed_builtins, local_vars)
        output = local_vars.get("result", None)
        return True, output
    except Exception as e:
        return False, traceback.format_exc()

def extract_text_from_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except:
        return ""

def clean_output(output):
    if not output:
        return ""
    output = output.strip()
    if output.startswith("```"):
        output = output.split("```")[1]
    output = output.replace("\n", "").replace("\r", "").strip()
    return output

def dataframe_to_base64_png(df, x_col, y_col):
    fig, ax = plt.subplots()
    ax.scatter(df[x_col], df[y_col])
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()
