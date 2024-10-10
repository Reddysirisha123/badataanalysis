import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('retaildataset/sample_customer_data_for_exam.csv')

# 1. Display the first few rows
print("First few rows of the dataset:")
print(df.head())

# 2. Create a heatmap to visualize correlation between numerical variables
plt.figure(figsize=(12, 10))
sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Numerical Variables')
plt.savefig('correlation_heatmap.png')
plt.close()

# 3. Create histograms for age and income
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.hist(df['age'], bins=20, edgecolor='black')
ax1.set_title('Distribution of Age')
ax1.set_xlabel('Age')
ax1.set_ylabel('Frequency')

ax2.hist(df['income'], bins=20, edgecolor='black')
ax2.set_title('Distribution of Income')
ax2.set_xlabel('Income')
ax2.set_ylabel('Frequency')

plt.tight_layout()
plt.savefig('age_income_histograms.png')
plt.close()

# 4. Generate a box plot for purchase amount across product categories
plt.figure(figsize=(12, 6))
sns.boxplot(x='product_category', y='purchase_amount', data=df)
plt.title('Distribution of Purchase Amount Across Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Purchase Amount')
plt.xticks(rotation=45)
plt.savefig('purchase_amount_boxplot.png')
plt.close()

# 5. Create a pie chart to visualize the proportion of customers by gender
gender_counts = df['gender'].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Proportion of Customers by Gender')
plt.axis('equal')
plt.savefig('gender_pie_chart.png')
plt.close()

print("Analysis complete. Please check the generated image files for visualizations.")

avg_purchase_by_education = df.groupby('education')['purchase_amount'].mean().sort_values(ascending=False)
print("Average Purchase Amount by Education Level:")
print(avg_purchase_by_education)
print()

# 2. Calculate average satisfaction score for each loyalty status
avg_satisfaction_by_loyalty = df.groupby('loyalty_status')['satisfaction_score'].mean().sort_values(ascending=False)
print("Average Satisfaction Score by Loyalty Status:")
print(avg_satisfaction_by_loyalty)
print()

# 3. Create bar plot comparing purchase frequency across different regions
plt.figure(figsize=(10, 6))
df['region'].value_counts().plot(kind='bar')
plt.title('Purchase Frequency Across Regions')
plt.xlabel('Region')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('purchase_frequency_by_region.png')
plt.close()

# 4. Compute percentage of customers who used promotional offers
promo_usage_percentage = (df['promotion_usage'].sum() / len(df)) * 100
print(f"Percentage of customers who used promotional offers: {promo_usage_percentage:.2f}%")
print()

# 5. Investigate correlation between income and purchase amount
correlation = df['income'].corr(df['purchase_amount'])
print(f"Correlation between income and purchase amount: {correlation:.2f}")

# Create a scatter plot to visualize the relationship
plt.figure(figsize=(10, 6))
plt.scatter(df['income'], df['purchase_amount'], alpha=0.5)
plt.title('Income vs Purchase Amount')
plt.xlabel('Income')
plt.ylabel('Purchase Amount')
plt.tight_layout()
plt.savefig('income_vs_purchase_amount.png')
plt.close()

print("\nAnalysis complete. Please check the generated image files for visualizations.")


# Read the CSV file
df = pd.read_csv('retaildataset/sample_customer_data_for_exam.csv')

# 1. Scatter plot of purchase frequency vs purchase amount, color-coded by loyalty status
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='purchase_frequency', y='purchase_amount', hue='loyalty_status', palette='deep')
plt.title('Purchase Frequency vs Purchase Amount by Loyalty Status')
plt.xlabel('Purchase Frequency')
plt.ylabel('Purchase Amount')
plt.legend(title='Loyalty Status')
plt.show()

# 2. Average purchase amount for customers who used promotions vs those who did not
avg_purchase = df.groupby('promotion_usage')['purchase_amount'].mean()
print("\nAverage Purchase Amount:")
print(avg_purchase)

# 3. Violin plot showing the distribution of satisfaction scores for different loyalty status groups
plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='loyalty_status', y='satisfaction_score')
plt.title('Distribution of Satisfaction Scores by Loyalty Status')
plt.xlabel('Loyalty Status')
plt.ylabel('Satisfaction Score')
plt.show()

# 4. Stacked bar chart showing the proportion of promotion usage across different product categories
promo_by_category = df.groupby('product_category')['promotion_usage'].value_counts(normalize=True).unstack()
promo_by_category.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Proportion of Promotion Usage by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Proportion')
plt.legend(title='Promotion Usage', labels=['No Promotion', 'Used Promotion'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 5. Correlation between satisfaction score and purchase frequency
correlation = df['satisfaction_score'].corr(df['purchase_frequency'])
print(f"\nCorrelation between satisfaction score and purchase frequency: {correlation:.2f}")

# Calculate correlation for different loyalty status groups
correlations_by_status = df.groupby('loyalty_status').apply(lambda x: x['satisfaction_score'].corr(x['purchase_frequency']))
print("\nCorrelations by loyalty status:")
print(correlations_by_status)

