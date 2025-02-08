import torch

def fit(model, optimizer, criterion, train_loader, test_loader, num_epochs, device):
    train_losses = []
    train_acc = []
    val_losses = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()
        t_loss = 0
        t_acc = 0
        cnt = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            t_acc += (torch.argmax(outputs, 1) == y).sum().item()
            cnt += len(y)
        t_loss /= len(train_loader)
        train_losses.append(t_loss)
        t_acc /= cnt
        train_acc.append(t_acc)

        model.eval()
        v_loss = 0
        v_acc = 0
        cnt = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                v_loss += loss.item()
                v_acc += (torch.argmax(outputs, 1)==y).sum().item()
                cnt += len(y)
        v_loss /= len(test_loader)
        val_losses.append(v_loss)
        v_acc /= cnt
        val_acc.append(v_acc)
        print(f"Epoch {epoch+1}/{num_epochs}, Train_Loss: {t_loss:.4f}, Train_Acc: {t_acc:.4f}, Validation Loss: {v_loss:.4f}, Val_Acc: {v_acc:.4f}")

    return train_losses, train_acc, val_losses, val_acc