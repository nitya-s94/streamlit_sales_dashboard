import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# CONFIG
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="🛒",
    layout="wide"
)

sns.set_theme(style="whitegrid")

DATA_DIR = Path("data")


# HELPERS
def show_plot(fig):
    st.pyplot(fig)
    plt.close(fig)


#LOAD DATA
@st.cache_data
def load_data():
    try:
        orders = pd.read_csv(DATA_DIR / "olist_orders_dataset.csv")
        items = pd.read_csv(DATA_DIR / "olist_order_items_dataset.csv")
        customers = pd.read_csv(DATA_DIR / "olist_customers_dataset.csv")
        products = pd.read_csv(DATA_DIR / "olist_products_dataset.csv")
        reviews = pd.read_csv(DATA_DIR / "olist_order_reviews_dataset.csv")

        return orders, items, customers, products, reviews

    except FileNotFoundError:
        st.error("Dataset files not found inside /data folder.")
        st.stop()


orders, items, customers, products, reviews = load_data()


#PREPROCESSING
orders["order_purchase_timestamp"] = pd.to_datetime(
    orders["order_purchase_timestamp"]
)

# Keep period datatype for proper sorting
orders["month"] = (
    orders["order_purchase_timestamp"]
    .dt.to_period("M")
)

# Main merged dataframe
df = (
    orders
    .merge(items, on="order_id")
    .merge(customers, on="customer_id")
)


# HEADER
st.title("🛒 E-Commerce Sales Dashboard")
st.markdown(
    "Analyzing 100,000+ real orders from Brazil’s Olist e-commerce platform"
)


# SIDEBAR
st.sidebar.header("🔍 Filters")

states = sorted(df["customer_state"].dropna().unique())
selected_state = st.sidebar.selectbox(
    "Select State",
    ["All"] + list(states)
)

years = sorted(
    orders["order_purchase_timestamp"]
    .dt.year
    .dropna()
    .unique()
)

selected_year = st.sidebar.selectbox(
    "Select Year",
    ["All"] + [str(y) for y in years]
)


# Apply filters
filtered_df = df.copy()

if selected_state != "All":
    filtered_df = filtered_df[
        filtered_df["customer_state"] == selected_state
    ]

if selected_year != "All":
    filtered_df = filtered_df[
        filtered_df["order_purchase_timestamp"].dt.year
        == int(selected_year)
    ]


#KPI SECTION
st.markdown("---")
st.subheader("📊 Key Metrics")

total_revenue = filtered_df["price"].sum()
total_orders = filtered_df["order_id"].nunique()
total_customers = filtered_df["customer_unique_id"].nunique()

avg_order_value = (
    total_revenue / total_orders
    if total_orders > 0 else 0
)

k1, k2, k3, k4 = st.columns(4)

k1.metric("💰 Total Revenue", f"${total_revenue:,.0f}")
k2.metric("📦 Orders", f"{total_orders:,}")
k3.metric("👥 Customers", f"{total_customers:,}")
k4.metric("🧾 Avg Order Value", f"${avg_order_value:,.2f}")


#CHART 1
st.markdown("---")
st.subheader("📈 Monthly Revenue Trend")

monthly = (
    filtered_df
    .groupby("month")["price"]
    .sum()
    .reset_index()
    .sort_values("month")
)

x = range(len(monthly))

fig, ax = plt.subplots(figsize=(12, 4))

ax.plot(
    x,
    monthly["price"],
    linewidth=2,
    marker="o",
    markersize=4
)

ax.fill_between(
    x,
    monthly["price"],
    alpha=0.15
)

ax.set_xticks(list(x))
ax.set_xticklabels(
    monthly["month"].astype(str),
    rotation=45
)

ax.set_xlabel("Month")
ax.set_ylabel("Revenue ($)")
ax.set_title("Monthly Revenue")

plt.tight_layout()
show_plot(fig)


#CHARTS 2,3
st.markdown("---")

left, right = st.columns(2)


# Top categories
with left:

    st.subheader("🏆 Top Categories")

    category_df = (
        filtered_df
        .merge(products, on="product_id")
        .groupby("product_category_name")["price"]
        .sum()
        .reset_index()
        .dropna()
        .sort_values("price", ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=category_df,
        x="price",
        y="product_category_name",
        ax=ax
    )

    ax.set_xlabel("Revenue ($)")
    ax.set_ylabel("")

    plt.tight_layout()
    show_plot(fig)


# Order status
with right:

    st.subheader("📦 Order Status")

    status_df = (
        filtered_df["order_status"]
        .value_counts()
        .reset_index()
    )

    status_df.columns = [
        "order_status",
        "count"
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.pie(
        status_df["count"],
        labels=status_df["order_status"],
        autopct="%1.1f%%"
    )

    ax.set_title("Order Status Distribution")

    plt.tight_layout()
    show_plot(fig)


#CHARTS 4,5
st.markdown("---")

left2, right2 = st.columns(2)


# Delivery analysis
with left2:

    st.subheader("🚚 Fastest Delivery States")

    delivery_df = (
        orders
        .merge(customers, on="customer_id")
        .copy()
    )

    delivery_df["order_delivered_customer_date"] = pd.to_datetime(
        delivery_df["order_delivered_customer_date"],
        errors="coerce"
    )

    # Remove incomplete deliveries
    delivery_df = delivery_df.dropna(
        subset=["order_delivered_customer_date"]
    )

    delivery_df["delivery_days"] = (
        delivery_df["order_delivered_customer_date"]
        - delivery_df["order_purchase_timestamp"]
    ).dt.days

    state_delivery = (
        delivery_df
        .groupby("customer_state")["delivery_days"]
        .mean()
        .reset_index()
        .sort_values("delivery_days")
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.barplot(
        data=state_delivery,
        x="delivery_days",
        y="customer_state",
        ax=ax
    )

    ax.set_xlabel("Days")
    ax.set_ylabel("")

    plt.tight_layout()
    show_plot(fig)


# Review analysis
with right2:

    st.subheader("⭐ Customer Reviews")

    review_df = reviews.merge(
        orders[["order_id"]],
        on="order_id"
    )

    fig, ax = plt.subplots(figsize=(7, 5))

    sns.countplot(
    data=review_df,
    x="review_score",
    hue="review_score",
    palette="YlOrRd",
    order=[1,2,3,4,5],
    legend=False,
    ax=ax
    )

    ax.set_xlabel("Review Score")
    ax.set_ylabel("Count")

    plt.tight_layout()
    show_plot(fig)


# METRIC
st.markdown("---")
st.subheader("💡 Customer Insight")

repeat_rate = (
    filtered_df
    .groupby("customer_unique_id")
    .size()
    .gt(1)
    .mean()
    * 100
)

st.metric(
    "Repeat Customer Rate",
    f"{repeat_rate:.1f}%"
)


#FOOTER
st.markdown("---")
st.caption(
    "Data Source: Olist Brazilian E-Commerce Dataset"
)