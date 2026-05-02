from flask import Flask, render_template, request

app = Flask(__name__)

# Menu data
menu = {
    "pizza": "Rs. 1200",
    "burger": "Rs. 500",
    "pasta": "Rs. 800",
    "shawarma": "Rs. 300"
}

# Order status (by item name)
orders = {
    "pizza": "Preparing",
    "burger": "Out for delivery",
    "pasta": "Delivered",
    "shawarma": "Preparing"
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot():
    user_message = request.form["msg"].lower()

    # 1. Greeting
    if "hi" in user_message or "hello" in user_message:
        return "Welcome! Ask me about menu, reservation, or order."

    # 2. Menu
    elif "menu" in user_message:
        response = "Menu:\n"
        for item, price in menu.items():
            response += f"{item.title()} - {price}\n"
        return response

    # 3. Food selection / ordering
    elif any(item in user_message for item in menu):
        for item in menu:
            if item in user_message:
                return f"You selected {item.title()} ({menu[item]}). Your order is being prepared!"

    # 4. Reservation
    elif ("reservation" in user_message or 
          "book" in user_message or 
          "table" in user_message or 
          "reserve" in user_message):
        return "Tables are available! You can book a table now."

    # 5. Order tracking
    elif "order" in user_message:
        for item in orders:
            if item in user_message:
                return f"Your {item.title()} order status: {orders[item]}"
        return "Order not found. Please mention item name (pizza, burger, etc.)."

    # 6. Default
    else:
        return "Sorry, I didn’t understand. Try 'menu', 'pizza', 'reservation', or 'order pizza'."

if __name__ == "__main__":
    app.run(debug=True)