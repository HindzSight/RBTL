print(df.label.value_counts())

# print(df.label.value_counts(normalize=True) * 100)

# fig, ax = plt.subplots(figsize=(8, 8))

# counts = df.label.value_counts(normalize=True) * 100

# sns.barplot(x=counts.index, y=counts, ax=ax)

# ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
# ax.set_ylabel("Percentage")

# plt.show()