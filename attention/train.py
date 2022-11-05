
def train(model, epochs=50, lr=0.001):

    # Define loss function and optimizer
    data_train = pd.DataFrame(columns=('step','loss','precision','recall','f1','tp','fp','fn'))
    data_val = pd.DataFrame(columns=('step','loss','precision','recall','f1','tp','fp','fn'))
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # main loop
    for epoch in range(epochs):
        # train
        model.train()
        losses = []
        tps, fps, fns = 0, 0, 0
        for i, data in enumerate(trainloader):
            # get the inputs
            inputs, labels = data
            # forward pass
            if model.encoder_name == 'GCN':
                adj = sequence2matrix(inputs, input_format == 'CGUA')
                outputs = model(inputs.to(device), adj.to(device))
            else:
                outputs = model(inputs.to(device))

            # compute and collect metrics
            labels_sym_matrix = sequence2matrix(labels)
            if outputs.size(-1) == 3:
                outputs_sym_matrix = sequence2matrix(
                    torch.argmax(outputs.detach().cpu(), -1))
                tp, fp, fn = compute_metrics(
                    outputs_sym_matrix, labels_sym_matrix)
            else:
                tp, fp, fn = compute_metrics(
                    outputs.detach().cpu(), labels_sym_matrix)

            tps += tp
            fps += fp
            fns += fn

            # compute loss
            if model.decoder_name == 'symmetric_matrix':
                labels = torch.argmax(labels_sym_matrix, -1)
            outputs, labels = outputs.view(-1, outputs.size(2)
                                           ), labels.view(-1).to(device)
            loss = loss_func(outputs, labels)
            losses.append(loss.item())

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # eval
        model.eval()
        valid_losses = []
        valid_tps, valid_fps, valid_fns = 0, 0, 0
        for i, data in enumerate(validloader):
            # get the inputs
            inputs, labels = data

            # forward pass
            with torch.no_grad():
                if model.encoder_name == 'GCN':
                    adj = sequence2matrix(inputs, input_format == 'CGUA')
                    outputs = model(inputs.to(device), adj.to(device))
                else:
                    outputs = model(inputs.to(device))

            # compute and collect metrics
            labels_sym_matrix = sequence2matrix(labels)
            if outputs.size(-1) == 3:
                outputs_sym_matrix = sequence2matrix(
                    torch.argmax(outputs.detach().cpu(), -1))
                tp, fp, fn = compute_metrics(
                    outputs_sym_matrix, labels_sym_matrix)
            else:
                tp, fp, fn = compute_metrics(
                    outputs.detach().cpu(), labels_sym_matrix)

            valid_tps += tp
            valid_fps += fp
            valid_fns += fn

            if model.decoder_name == 'symmetric_matrix':
                labels = torch.argmax(labels_sym_matrix, -1)
            outputs, labels = outputs.view(-1, outputs.size(2)
                                           ), labels.view(-1).to(device)
            loss = loss_func(outputs, labels)
            valid_losses.append(loss.item())

        # print out results every epoch
        precision, recall, f1 = compute_f1(tps, fps, fns)
        valid_precision, valid_recall, valid_f1 = compute_f1(
            valid_tps, valid_fps, valid_fns)
        print('Epoch: {}, train loss:{}, precision:{}, recall:{}, f1:{}'.format(
            epoch,
            round(statistics.mean(losses), 2),
            round(precision, 2), round(recall, 2), round(f1, 2)
        ))
        print('Epoch: {}, val   loss:{}, precision:{}, recall:{}, f1:{}'.format(
            epoch,
            round(statistics.mean(valid_losses), 2),
            round(valid_precision, 2), round(
                valid_recall, 2), round(valid_f1, 2)
        ))

        data_train.loc[epoch] = [epoch,round(statistics.mean(losses), 2),round(precision, 2),
                             round(recall, 2), round(f1, 2),tps, fps, fns]
        data_val.loc[epoch] = [epoch,round(statistics.mean(valid_losses), 2),round(valid_precision, 2),
                             round(valid_recall, 2), round(valid_f1, 2),valid_tps, valid_fps, valid_fns]


    return data_train, data_val

# Prepare encoder keyword arguments
rnn_kwargs = {'lstm_layers': 2, 'lstm_dropout': 0.5,
              'use_attention': False, 'is_final_output': False}
gcn_kwargs = {'num_embed': 4,
              'embed_dim': 32,
              'embed_dropout': 0,
              'hidden_dims': [128, 128],
              'dropout': 0.5}

# Create a model and train
encoder = 'RNN'
decoder = 'symmetric_matrix'

if encoder == 'RNN':
    encoder_kwargs = rnn_kwargs
if encoder == 'GCN':
    encoder_kwargs = gcn_kwargs

model = RNAStructNet(encoder=encoder,
                     decoder=decoder,
                     **encoder_kwargs).to(device)
data_train, data_val = train(model, epochs=20, lr=0.005)
