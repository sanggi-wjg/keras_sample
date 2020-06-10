import matplotlib.pyplot as plt

math_scores = [50, 60, 70, 80, 90, 85, 90, 95]
eng_scores = [80, 70, 90, 90, 80, 80, 90, 100]
sample_range = range(len(math_scores))

plt.plot(sample_range, math_scores, 'b', label = 'Math')
plt.plot(sample_range, eng_scores, 'g', label = 'English')
plt.ylabel('Score')
plt.xlabel('Times')
plt.legend()
plt.show()
