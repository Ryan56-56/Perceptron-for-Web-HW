import random

GRID_SIZE = 4
SAMPLE_COUNT = 120
LABEL_NAMES = {1: 'L', -1: 'T'}

class Perceptron:
    def __init__(self, size):
        self.weights = [0] * size
        self.bias = 0

    def predict_raw(self, inputs):
        s = self.bias
        for i in range(len(self.weights)): # Use self.weights for length
            s += self.weights[i] * inputs[i]
        return s

    def predict(self, inputs):
        return 1 if self.predict_raw(inputs) >= 0 else -1

    def train(self, samples, max_epochs=1000):
        epoch = 0
        while epoch < max_epochs:
            errors = 0
            random.shuffle(samples)
            for sample in samples:
                activation = self.predict_raw(sample['pixels'])
                if sample['label'] * activation <= 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += sample['label'] * sample['pixels'][i]
                    self.bias += sample['label']
            if errors == 0:
                break
            epoch += 1
        return epoch + 1

def build_dataset(sample_count):
    dataset = []
    each_kind = round(sample_count / 2)
    for _ in range(each_kind):
        dataset.append({'pixels': make_pattern('L'), 'label': 1})
        dataset.append({'pixels': make_pattern('T'), 'label': -1})
    return dataset

def make_pattern(kind):
    if kind == 'L':
        base = [1, 0, 0, 0,
                1, 0, 0, 0,
                1, 0, 0, 0,
                1, 1, 1, 0]
        noise_positions = [6, 7, 10, 11, 15]
    else: # kind == 'T'
        base = [1, 1, 1, 1,
                0, 1, 0, 0,
                0, 1, 0, 0,
                0, 1, 0, 0]
        noise_positions = [6, 7, 10, 11, 14, 15]

    pattern = base.copy()
    for index in noise_positions:
        if random.random() < 0.28:
            pattern[index] = 1
    return pattern

def evaluate(model, samples):
    correct = 0
    for sample in samples:
        prediction = model.predict(sample['pixels'])
        if prediction == sample['label']:
            correct += 1
    return round((correct / len(samples)) * 100)

def label_text(label):
    return LABEL_NAMES.get(label, 'unknown')

# --- Demonstration --- 
dataset = build_dataset(SAMPLE_COUNT)
perceptron = Perceptron(GRID_SIZE * GRID_SIZE)
epochs = perceptron.train(dataset, 2000)
accuracy = evaluate(perceptron, dataset)

print(f"Trained on {len(dataset)} labeled samples. Accuracy: {accuracy}% after {epochs} epochs.")

# --- Example prediction --- 
print("\n--- Example Predictions ---")

# Predict 'L' pattern
l_pattern = make_pattern('L')
print(f"'L' Pattern: {l_pattern}")
print(f"Predicted label for 'L' pattern: {label_text(perceptron.predict(l_pattern))}")

# Predict 'T' pattern
t_pattern = make_pattern('T')
print(f"'T' Pattern: {t_pattern}")
print(f"Predicted label for 'T' pattern: {label_text(perceptron.predict(t_pattern))}")

# Predict a random sample from the dataset
random_sample = random.choice(dataset)
print(f"\nRandom sample pixels: {random_sample['pixels']}")
print(f"Actual label: {label_text(random_sample['label'])}")
print(f"Predicted label: {label_text(perceptron.predict(random_sample['pixels']))}")
