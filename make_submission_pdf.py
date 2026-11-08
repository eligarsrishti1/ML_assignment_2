"""
make_submission_pdf.py - builds the single submission PDF in the required order:
  1. GitHub link  2. Streamlit link  3. Virtual Lab screenshot  4. README content

Usage:
  python make_submission_pdf.py --github https://github.com/you/repo \
      --streamlit https://yourapp.streamlit.app --screenshot lab.png
Needs: pip install reportlab
"""
import argparse
import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ap = argparse.ArgumentParser()
ap.add_argument("--github", default="<PASTE GITHUB REPO LINK HERE>")
ap.add_argument("--streamlit", default="<PASTE STREAMLIT APP LINK HERE>")
ap.add_argument("--screenshot", default=None)
ap.add_argument("--name", default="<Your Name>")
ap.add_argument("--id", default="<Your BITS ID>")
ap.add_argument("--out", default="Assignment2_Submission.pdf")
a = ap.parse_args()

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=17, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=5)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
C = ParagraphStyle("C", parent=B, fontSize=8.3, leading=10.5)
CH = ParagraphStyle("CH", parent=C, textColor=colors.white, fontName="Helvetica-Bold")


def link(url):
    from html import escape
    return f'<link href="{escape(url)}" color="blue">{escape(url)}</link>'


def table(data, widths, header=True):
    t = Table(data, colWidths=widths, repeatRows=1)
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
          ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("FONTSIZE", (0, 0), (-1, -1), 8.5)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
               ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
               ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(st))
    return t


def P(txt, s=C):
    return Paragraph(txt, s)


m = pd.read_csv("model/metrics.csv")
story = [
    Paragraph("Machine Learning – Assignment 2", H1),
    Paragraph("M.Tech (AIML) · Work Integrated Learning Programmes Division, BITS Pilani", B),
    Paragraph(f"Name: {a.name} &nbsp;&nbsp; BITS ID: {a.id}", B),
    Spacer(1, 10),
    Paragraph("1. GitHub Repository Link", H2),
    Paragraph(link(a.github), B),
    Paragraph("Contains: app.py, train_models.py, requirements.txt, README.md, test_data.csv, model/ (5 saved models).", B),
    Paragraph("2. Live Streamlit App Link", H2),
    Paragraph(link(a.streamlit), B),
    Paragraph("Deployed on Streamlit Community Cloud.", B),
    Paragraph("3. Screenshot – Assignment execution on BITS Virtual Lab", H2),
]
if a.screenshot and os.path.exists(a.screenshot):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(a.screenshot).getSize()
    mw = 17 * cm
    sc = min(mw / w, 12 * cm / h)
    story.append(Image(a.screenshot, w * sc, h * sc))
else:
    box = Table([[P("<br/><br/>[ Insert ONE screenshot of the assignment executed on BITS Virtual Lab here ]<br/><br/><br/>", B)]],
                colWidths=[17 * cm])
    box.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 1, colors.grey)]))
    story.append(box)

story += [PageBreak(), Paragraph("4. README.md Content", H1),
          Paragraph("a) Problem Statement", H2),
          Paragraph("Early and accurate diagnosis of breast cancer is critical. This project builds, evaluates and compares five classification "
                    "models that predict whether a breast tumour is <b>malignant</b> or <b>benign</b> from measurements computed on a digitised image of a "
                    "fine-needle aspirate (FNA). The models are deployed in an interactive Streamlit app that evaluates each saved model on uploaded test data.", B),
          Paragraph("b) Dataset Description", H2)]

ds = [["Property", "Details"],
      ["Dataset", "Breast Cancer Wisconsin (Diagnostic) – WDBC"],
      ["Source", "UCI ML Repository (bundled with scikit-learn: load_breast_cancer)"],
      ["Problem type", "Binary classification"],
      ["Instances / Features", "569 instances (min 500) / 30 numeric features (min 12)"],
      ["Target", "0 = Malignant (212), 1 = Benign (357)"],
      ["Missing values", "None"],
      ["Split", "80/20 stratified, random_state=42 → 455 train / 114 test"],
      ["Pre-processing", "StandardScaler inside Pipeline for LR, KNN, SVM; none for Decision Tree and Naive Bayes"]]
story += [table([[P(c, CH if i == 0 else C) for c in r] for i, r in enumerate(ds)], [4 * cm, 13 * cm]),
          Spacer(1, 4),
          Paragraph("The 30 features are the mean, standard error and worst value of 10 nucleus characteristics: radius, texture, perimeter, area, "
                    "smoothness, compactness, concavity, concave points, symmetry, fractal dimension.", B),
          Paragraph("GitHub Repository Link: " + link(a.github), B),
          Paragraph("c) Models Used – Comparison Table", H2)]

rows = [list(m.columns)] + [[r[0]] + [f"{v:.4f}" for v in r[1:]] for r in m.values.tolist()]
story += [table(rows, [4.6 * cm] + [2.07 * cm] * 6),
          Spacer(1, 3),
          Paragraph("Precision/Recall/F1 are for the positive class (Benign). Hyper-parameters: LR (max_iter=1000), DT (max_depth=5), KNN (k=5), "
                    "GaussianNB (defaults), SVM (RBF, probability=True).", C),
          Paragraph("d) Observations on Model Performance", H2)]

obs = [
    ["Logistic Regression", "Joint-best (Acc 98.25%, AUC 0.9954, MCC 0.9623). Classes are almost linearly separable after standardisation, so a simple, fast, interpretable linear model generalises very well."],
    ["Decision Tree", "Weakest (Acc 92.11%, AUC 0.9163, MCC 0.8341). A single depth-5 tree has high variance and poorly ranked probabilities (lowest AUC); recall 0.9167 means more benign cases missed."],
    ["KNN", "Good, balanced (Acc 95.61%, AUC 0.9788, MCC 0.9054). Needs scaling and is sensitive to k and noise in 30 dimensions; a bit below LR/SVM."],
    ["Naive Bayes", "Accuracy 93.86% but high AUC 0.9878: ranks cases well but the default threshold is not optimal. Independence assumption is violated by correlated features (radius/perimeter/area); MCC 0.8676."],
    ["Support Vector Machine", "Ties with LR on Accuracy/F1/MCC (98.25%, 0.9861, 0.9623), AUC 0.9950. RBF kernel captures non-linearity; slower and less interpretable than LR."],
    ["Overall, Winner for your dataset?", "<b>Logistic Regression</b> – ties with SVM on Accuracy/F1/MCC, marginally highest AUC, and simpler/faster/more interpretable. SVM is a very close runner-up. With 114 test samples, one sample ≈ 0.9% accuracy, so read the top-two ranking with caution."],
]
od = [["ML Model Name", "Observation about model performance"]] + obs
story += [table([[P(c, CH if i == 0 else C) for c in r] for i, r in enumerate(od)], [4 * cm, 13 * cm])]

SimpleDocTemplate(a.out, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                  title="ML Assignment 2 Submission").build(story)
print("Created", a.out)
