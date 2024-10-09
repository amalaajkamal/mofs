import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Define the Comparative Neural Network (CmpNN)
class ComparativeNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers):
        super(ComparativeNeuralNetwork, self).__init__()
        self.hidden_layers = nn.ModuleList()
        
        # Input Layer to first hidden layer
        self.hidden_layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Add additional hidden layers
        for i in range(1, len(hidden_layers)):
            self.hidden_layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
        
        # Two output nodes for comparative ranking (x < y and x > y)
        self.output_layer = nn.Linear(hidden_layers[-1], 2)

    def forward(self, x):
        # Forward pass through the hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        # Output layer (logits for the two classes)
        x = self.output_layer(x)
        
        # Apply log_softmax over the correct dimension (dim=0 for 1D tensor)
        return F.log_softmax(x, dim=0)

# Data Preparation
def prepare_data(doc_pairs, labels):
    # doc_pairs: List of tuples [(doc1, doc2)]
    # labels: Corresponding list of labels (0 for doc1 < doc2, 1 for doc1 > doc2)
    data = []
    for (doc1, doc2), label in zip(doc_pairs, labels):
        concat_docs = torch.cat([doc1, doc2], dim=0)  # Concatenate the feature vectors of the two documents
        data.append((concat_docs, torch.tensor([label], dtype=torch.long)))
    return data

# Training the Model
def train_model(model, data, epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for inputs, label in data:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output.unsqueeze(0), label)  # Ensure output and label are 2D tensors
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data)}")

# Rank Aggregation (Simple bagging of multiple models)
def rank_aggregation(models, doc_pairs):
    rankings = []
    
    for doc1, doc2 in doc_pairs:
        votes = torch.zeros(2)  # To store votes for (x < y) and (x > y)
        for model in models:
            concat_docs = torch.cat([doc1, doc2], dim=0)
            output = model(concat_docs)
            prediction = torch.argmax(output)
            votes[prediction] += 1
        
        rankings.append(torch.argmax(votes).item())
    
    return rankings

# Example usage
if __name__ == "__main__":
    input_size = 100  # Example input size (feature vector size of each document)
    hidden_layers = [50, 25]  # Example hidden layers

    # Initialize the model
    model = ComparativeNeuralNetwork(input_size * 2, hidden_layers)

    # Example data (each document represented by a 100-dimensional vector)
    doc_pairs = [(torch.randn(100), torch.randn(100)) for _ in range(1000)]
    labels = [0 if torch.rand(1).item() < 0.5 else 1 for _ in range(1000)]

    # Prepare data
    data = prepare_data(doc_pairs, labels)

    # Train the model
    train_model(model, data, epochs=100)

    # Perform rank aggregation using multiple models
    models = [model]  # Example of using only one model; you can use multiple models for rank aggregation
    final_rankings = rank_aggregation(models, doc_pairs)
    print("Final rankings:", final_rankings)
