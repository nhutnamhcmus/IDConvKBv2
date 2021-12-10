from config import Trainer, Tester
from module.model import IDConvKB
from module.loss import SigmoidLoss
from module.strategy import NegativeSampling
from data import TrainDataLoader, TestDataLoader

# dataloader for training
train_dataloader = TrainDataLoader(
	in_path = "./benchmarks/WN18RR/", 
	batch_size = 2048,
	threads = 32,
	sampling_mode = "cross", 
	bern_flag = 0, 
	filter_flag = 1, 
	neg_ent = 64,
	neg_rel = 0
)

# dataloader for test
test_dataloader = TestDataLoader("./benchmarks/WN18RR/", "link")

# define the model
convkb = IDConvKB(
	ent_tot = train_dataloader.get_ent_tot(),
	rel_tot = train_dataloader.get_rel_tot(),
	dim = 100,
	#margin = 6.0,
	#epsilon = 2.0,
)

# define the loss function
model = NegativeSampling(
	model = convkb, 
	loss = SigmoidLoss(adv_temperature = 2),
	batch_size = train_dataloader.get_batch_size(), 
	regul_rate = 1.0
)

# train the model
trainer = Trainer(model = model, data_loader = train_dataloader, train_times = 1, alpha = 2e-5, use_gpu = True, opt_method = "adam")
trainer.run()
convkb.save_checkpoint('./checkpoint/rotate_WN18RR_adv.ckpt')

# test the model
convkb.load_checkpoint('./checkpoint/rotate_WN18RR_adv.ckpt')
tester = Tester(model = convkb, data_loader = test_dataloader, use_gpu = True)
tester.run_link_prediction(type_constrain = True)