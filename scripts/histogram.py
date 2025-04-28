import matplotlib.pyplot as plt

# So sánh phân bố của haz_future
plt.hist(y_train_male, bins=30, alpha=0.5, label='Train (Nam)')
plt.hist(y_test_male, bins=30, alpha=0.5, label='Test (Nam)')
plt.xlabel('haz_future')
plt.ylabel('Số lượng')
plt.legend()
plt.title('Phân bố haz_future (Nam)')
plt.show()

# Tương tự cho nữ
plt.hist(y_train_female, bins=30, alpha=0.5, label='Train (Nữ)')
plt.hist(y_test_female, bins=30, alpha=0.5, label='Test (Nữ)')
plt.xlabel('haz_future')
plt.ylabel('Số lượng')
plt.legend()
plt.title('Phân bố haz_future (Nữ)')
plt.show()

# So sánh phân bố của age_future
plt.hist(X_train_male['age_future'], bins=30, alpha=0.5, label='Train (Nam)')
plt.hist(X_test_male['age_future'], bins=30, alpha=0.5, label='Test (Nam)')
plt.xlabel('age_future')
plt.ylabel('Số lượng')
plt.legend()
plt.title('Phân bố age_future (Nam)')
plt.show()

plt.hist(X_train_female['age_future'], bins=30, alpha=0.5, label='Train (Nữ)')
plt.hist(X_test_female['age_future'], bins=30, alpha=0.5, label='Test (Nữ)')
plt.xlabel('age_future')
plt.ylabel('Số lượng')
plt.legend()
plt.title('Phân bố age_future (Nữ)')
plt.show()