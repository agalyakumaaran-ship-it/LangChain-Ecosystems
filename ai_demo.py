
import smtplib
from email.message import EmailMessage

sender_email = "agalyakumaaran15@gmail.com"
app_password = "fbgb tjhv aums jjvf"
receiver_email = "agalyakumaaran15@gmail.com"

msg = EmailMessage()
msg.set_content(
    "Hello! This is a test email sent from my Python automation script."
)
msg["Subject"] = "Test Email from Python"
msg["From"] = sender_email
msg["To"] = receiver_email


try:
    print("Connecting to Gmail SMTP server...")

    with smtplib.SMTP_SSL(
        "smtp.gmail.com", 465, timeout=30
    ) as smtp:
        print("Logging in...")
        smtp.login(sender_email, app_password)

        print("Sending message...")
        smtp.send_message(msg)

    print("Email sent successfully!")

except Exception as e:
    print(f"FAILED with error: {e}")
