def train_test_MGDA(model, data_name, mod_params_mgda, device):
    model_repetitions = mod_params_mgda["model_repetitions"]
    training_epochs = mod_params_mgda["training_epochs"]
    archi = mod_params_mgda["archi"]
    batch_size = mod_params_mgda["batch_size"]
    img_shp = mod_params_mgda["img_shp"]
    momentum = mod_params_mgda["momentum"]
    lr = mod_params_mgda["lr"]
    model_dir_path = mod_params_mgda["model_dir_path"]
    
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)

    if data_name == "Cifar10Mnist":
        X_train, X_test, y_train, y_test = load_Cifar10Mnist_mgda()
    elif data_name == "MultiMnist":
        X_train, X_test, y_train, y_test = load_MultiMnist_mgda()
    else: raise ValueError(f"Unknown dataset {data_name} !")
    
    train_losses = []

    # Testing stuff
    test_accuracies = []

    from time import time
    # Start timer
    import datetime
    print(datetime.datetime.now())
    t0 = time()

    # dd/mm/YY H:M:S
    dt_string = datetime.datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
    model_dir_path = model_dir_path + "/" + dt_string
    os.mkdir(model_dir_path)
    
    model_multi = model.to(device)

    for i in range(model_repetitions):
        t_0_rep = time()
        print(f"######## Repetition {i+1}/{model_repetitions} ########")
        # model_multi = model
        loss_fn = nn.CrossEntropyLoss()
        MTLOptimizerClass = build_MGDA_optimizer(torch.optim.SGD)
        mtl_optim = MTLOptimizerClass(model_multi.parameters(), lr=lr, momentum=momentum)

        for epoch in tqdm(range(training_epochs)):
            #print("Training...")
            model_multi.train()
            train_loss = train_multi(torch.tensor(X_train, dtype=torch.float32, device=device),
                                                          torch.tensor(y_train, dtype=torch.float32, device=device),
                                                          model_multi, mtl_optim,
                                                          loss_fn, batch_size, device=device, archi = archi, img_shp=img_shp)
            train_losses.extend(train_loss)

            # Halve learning rate every 30 epochs
            if epoch > 0 and epoch % 30 == 0:
                for optim_param in mtl_optim.param_groups:
                    optim_param['lr'] = optim_param['lr'] / 2

        # Save model iteration
        model_multi.save_model(model_dir_path + "/" + f"model_{i}")
        
        T_norm_1_rep = time()-t_0_rep
        # Print computation time
        print('\nComputation time for one ITERATION: {} minutes'.format(T_norm_1_rep/60))

        print("Testing...")
        model_multi.train(mode=False)
        test_acc_task = test_multi(torch.tensor(X_test, dtype=torch.float32, device=device),
                                                              torch.tensor(y_test, dtype=torch.float32, device=device),
                                                              model_multi, loss_fn, batch_size,
                                                              device=device, img_shp = img_shp)

        test_accu_t1 = 100*sum(test_acc_task[0])/len(test_acc_task[0])
        test_accu_t2 = 100*sum(test_acc_task[1])/len(test_acc_task[1])
        test_accuracies.append([test_accu_t1, test_accu_t2])

        print(f"Finised Repetition {i+1} with Accuracy Task 1: {test_accu_t1}")
        print(f"Finised Repetition {i+1} with Accuracy Task 2: {test_accu_t2}")

    mean_acc = np.array(test_accuracies).mean(axis = 0)
    print("Mean Accuracy Task 1: ", mean_acc[0])
    print("Mean Accuracy Task 2: ", mean_acc[1])
    T_norm_1 = time()-t0
    # Print computation time
    print('\nComputation time: {} minutes'.format(T_norm_1/60))
    print(datetime.datetime.now())

    return train_losses, test_accuracies
