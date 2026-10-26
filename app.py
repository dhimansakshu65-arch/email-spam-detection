
import io
import urllib.request
import zipfile

import gradio as gr
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download the SMS spam dataset
url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
request = urllib.request.Request(
    url, headers={"User-Agent": "Mozilla/5.0"}
)

with urllib.request.urlopen(request, timeout=45) as response:
    data = response.read()

with zipfile.ZipFile(io.BytesIO(data)) as archive:
    with archive.open("SMSSpamCollection") as file:
        df = pd.read_csv(
            file, sep="\t", header=None,
            names=["label", "message"], encoding="utf-8"
        )

X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)

vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_vec, y_train)

accuracy = accuracy_score(
    y_test, model.predict(X_test_vec)
) * 100

def detect_spam(message):
    if not message or not message.strip():
        return "Please enter a message."

    prediction = model.predict(
        vectorizer.transform([message])
    )[0]

    if prediction == "spam":
        return "SPAM DETECTED - This message may be unwanted."
    return "MESSAGE APPEARS NORMAL - No spam detected."

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Email Spam Detection System"
) as demo:
    gr.Markdown(
        "# Email Spam Detection System\n"
        "### AI Applications Lab Project\n"
        "Enter a message below to check for spam."
    )

    message = gr.Textbox(
        label="Enter your message",
        placeholder="Type a message here...",
        lines=5
    )

    check = gr.Button("Check Message", variant="primary")
    result = gr.Textbox(label="Detection Result")

    check.click(
        fn=detect_spam,
        inputs=message,
        outputs=result
    )

    gr.Markdown(
        f"**Dataset:** {len(df):,} messages  \n"
        f"**Model:** TF-IDF + Multinomial Naive Bayes  \n"
        f"**Test accuracy:** {accuracy:.2f}%"
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
