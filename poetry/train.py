import json
import torch
from torch.utils.data import Dataset, random_split
import torch.utils.data
from models import *
from utils import *


train_loader = torch.utils.data.DataLoader(Dataset(),
                                           batch_size = 100,
                                           shuffle=True,
                                           pin_memory=True)
train_size=int(len(train_loader)*0.6)
val_size=int(len(train_loader)*0.2)
test_size=len(train_loader)-train_size-val_size
train_loader, val_loader, test_size = random_split(train_loader,[train_size, val_size, test_size])

d_model = 512
heads = 8
num_layers = 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 10

with open('WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)

transformer = Transformer(d_model = d_model, heads = heads, num_layers = num_layers, word_map = word_map)
transformer = transformer.to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size = d_model, warmup_steps = 4000, optimizer = adam_optimizer)
criterion = LossWithLS(len(word_map), 0.1).to(device)



def train(train_loader, transformer, criterion, epoch):

    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):

        samples = question.shape[0]

        # Move to device
        question = question.to(device)
        reply = reply.to(device)

        # Prepare Target Data
        # [batch_size, max_len]
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        # Create mask and add dimensions
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        # Get the transformer outputs
        out = transformer(question, question_mask, reply_input, reply_input_mask)

        # Compute the loss
        loss = criterion(out, reply_target, reply_target_mask)

        # Backprop
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()

        sum_loss += loss.item() * samples
        count += samples

        if i % 100 == 0:
            print("Epoch [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(train_loader), sum_loss/count))

def validate(val_loader, transformer, criterion, epoch):

    transformer.eval()
    sum_loss = 0
    count = 0

    with torch.no_grad():
        for i, (question, reply) in enumerate(val_loader):

            samples = question.shape[0]

            # Move to device
            question = question.to(device)
            reply = reply.to(device)

            # Prepare Target Data
            # [batch_size, max_len]
            reply_input = reply[:, :-1]
            reply_target = reply[:, 1:]

            # Create mask and add dimensions
            question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

            # Get the transformer outputs
            out = transformer(question, question_mask, reply_input, reply_input_mask)

            # Compute the loss
            loss = criterion(out, reply_target, reply_target_mask)

            sum_loss += loss.item() * samples
            count += samples

            if i % 100 == 0:
                print("Validation [{}][{}/{}]\tLoss: {:.3f}".format(epoch, i, len(val_loader), sum_loss / count))


for epoch in range(epochs):

    train(train_loader, transformer, criterion, epoch)
    validate(val_loader, transformer, criterion, epoch)

    state = {'epoch': epoch, 'transformer': transformer, 'transformer_optimizer': transformer_optimizer}

    torch.save(state, 'checkpoint_' + str(epoch) + '.pth.tar')
