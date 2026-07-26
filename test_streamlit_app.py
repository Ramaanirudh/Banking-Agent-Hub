from streamlit.testing.v1 import AppTest
import time

print("Starting Streamlit AppTest...")
at = AppTest.from_file("app.py", default_timeout=30)
at.run()

print("\n--- TEST B: Clarification Context ---")
at.chat_input[0].set_value("segment customers")
at.run(timeout=60)
print(f"Agent Reply 1: {at.chat_message[-1].markdown[0].value}")

at.chat_input[0].set_value("balance")
at.run(timeout=60)
print(f"Agent Reply 2: {at.chat_message[-1].markdown[0].value}")

print("\n--- TEST C: Out-of-scope ---")
at.chat_input[0].set_value("What's the best pizza topping?")
at.run(timeout=60)
print(f"Agent Reply 3: {at.chat_message[-1].markdown[0].value}")

print("\nAll automated tests completed.")
