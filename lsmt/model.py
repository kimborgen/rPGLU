
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dr_lstm, dr_fc, num_timesteps):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_timesteps = num_timesteps
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, dropout=dr_lstm)
        self.dropout = nn.Dropout(dr_fc)
        si = hidden_size * num_timesteps
        self.fc = nn.Linear(si, si * 2)
        self.fc2 = nn.Linear(si * 2, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        # x.size(1) is batch_size
        h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape ( seq_length, batch size, hidden_size)
        #last = out[-1, :, :]
        # Decode the hidden state of the last time step
        out_reshaped = out.transpose(0,1)
        out_reshaped2 = out_reshaped.reshape(out.size(1), -1)
        d1 = self.dropout(out_reshaped2)
        fc1 = self.fc(d1)
        r1 = F.relu(fc1)
        d2 = self.dropout(r1)
        fc2 = self.fc2(d2)

        return fc2


def train_lstm(datasets):

    experiment_name = "3layers, hs 640, dr_lstm 0.1, epoch 150, bs 128, all_hs 3fc * 12"
    input_size = 10
    num_classes = 9
    hidden_size = 635
    num_layers = 3
    lr=0.002
    dr_lstm = 0.1
    dr_fc = 0.3
    batch_size = 128
    num_epochs = 100

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    net = LSTMNet(input_size, hidden_size, num_layers, num_classes, dr_lstm, dr_fc, 23).to(device)
    print_params(net)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  

    

    train_iter = DataIterator(datasets.X_train, datasets.y_train, batch_size, device, shuffle_on_reset=True)
    #test_iter = DataIterator(datasets.X_val, datasets.y_val, batch_size, device)
    
    plots = TrainingPlots(seq=len(train_iter), experiment_name=experiment_name)    

    try:
        for epoch in range(num_epochs):
            
            if epoch != 0:
                train_iter.reset()
                #scheduler.step()
            with tqdm(total=len(train_iter), desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True) as pbar:
                
                for i, (X, y) in enumerate(train_iter):
                    pbar.set_postfix_str(f"i={i}")
                    pbar.update(1)

                    net.train()
                    #out_sum = ev_net_stepped(X, net)
                    out = ev_net_full(X, net)
                    loss = loss_fn(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                    optimizer.step()
                    

                    plots.update_loss(loss.item())
                    #print("loss: ", loss.item())
                    #loss_hist.append(loss.item())
                    plots.update_gradients(net)

                accuracy, precision, recall, f1 = evaluate_model(net, datasets, device, ev_net_full)
                print(f"Validation - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")
                plots.update_accuracy_metrics(accuracy, precision, recall, f1)
                plots.plot_all()
    except KeyboardInterrupt:
        print("hello there")
        pass

    #plt.show()
            
    # test
    input("Press Enter to continue...")
    input("Press Enter to continue...")

    plt.show()
    accuracy, precision, recall, f1 = test_model(net, datasets, device, ev_net_full)
    print(f"Test - Acc: {accuracy}, Prec: {precision}, Recall: {recall}, F1: {f1}")

    torch.save(net.state_dict(), "models/LSTM_baseline_1.pth")

    plt.show()
    
    input("Press Enter to continue...")
    input("Press Enter to continue...")
    input("Press Enter to continue...")
    input("Press Enter to continue...")