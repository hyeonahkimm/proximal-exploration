import time
import numpy as np
import pandas as pd
import wandb
import itertools

from lib.utils.dataset import BioSeqDataset
from utils.seq_utils import hamming_distance

from polyleven import levenshtein


def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(hamming_distance(*pair))
    return np.mean(dists)


def mean_distances_from_wt(seqs, wt):
    dists = []
    for seq in seqs:
        dists.append(hamming_distance(seq, wt))
    return np.mean(dists)


def mean_novelty(seqs, ref_seqs):
    novelty = [min([hamming_distance(seq, ref) for ref in ref_seqs]) for seq in seqs]
    return np.mean(novelty)


def pearson_correlation(X, Y):
    n = len(X)
    mean_X = sum(X) / n
    mean_Y = sum(Y) / n
    covariance = sum((X_i - mean_X) * (Y_i - mean_Y) for X_i, Y_i in zip(X, Y))
    std_X = np.sqrt(sum((X_i - mean_X)**2 for X_i in X))
    std_Y = np.sqrt(sum((Y_i - mean_Y)**2 for Y_i in Y))
    return covariance / (std_X * std_Y)


def spearman_correlation(X, Y):
    rank_X = np.argsort(np.argsort(-1 * X))  # Involves sorting - O(n log n)
    rank_Y = np.argsort(np.argsort(-1 * Y))  # Involves sorting - O(n log n)
    return pearson_correlation(rank_X, rank_Y)


def compute_correlations(seqs, scores, dist, proxy):
    print("Computing correlations")
    
    batch_size = 256
    
    if len(seqs) > 300:
        ys = []
        for i in range(len(seqs) // batch_size + 1):
            start, end = batch_size*i, min(batch_size*(i+1), len(seqs))
            if start == end:
                continue
            y = proxy.get_fitness(seqs[start:end])
            ys.extend(y.tolist())
        proxy_scores = np.array(ys)
    else:
        proxy_scores = proxy.get_fitness(seqs)
    
    rho = spearman_correlation(scores, proxy_scores)
    mse = np.mean((scores - proxy_scores)**2)
    
    print(f"Spearman: {rho:.3f}")
    print(f"MSE: {mse:.3f}")
    
    rst = {'proxy_scores': proxy_scores,
           'scores': scores, 
           'dist': dist,
           'rho_all': rho, 
           'mse_all': mse}
    
    if dist is not None:
        for th in np.arange(1, 11):
            neighbors_score, neighbors_proxy = scores[dist<=th], proxy_scores[dist<=th]
            if sum(dist<=th) < 1:
                continue
            neighbor_rho = spearman_correlation(neighbors_score, neighbors_proxy)
            neighbor_mse = np.mean((neighbors_score - neighbors_proxy)**2)
            
            print(f"Threshold: {th}")
            print(f"Num neighbors: {sum(dist<=th)}")
            print(f"Spearman: {neighbor_rho:.3f}")
            print(f"MSE: {neighbor_mse:.3f}")
            
            rst['rho'+str(th)] = neighbor_rho
            rst['mse'+str(th)] = neighbor_mse
    
    return rst


def evaluate_proxy(model, x_init, y_init, args):
    rst = compute_correlations(x_init, y_init, None, model)
    
    with open(f'./results/{args.task}_{args.net}_init_{args.seed}.npy', 'wb') as f:
        np.save(f, rst)
    
    x_heldout = np.load(f'./dataset/{args.task}/{args.task}-x-heldout3.npy', allow_pickle=True)[:256]
    y_heldout = np.load(f'./dataset/{args.task}/{args.task}-y-heldout3.npy', allow_pickle=True)[:256]
    dist = np.load(f'./dataset/{args.task}/{args.task}-dist-heldout3.npy', allow_pickle=True)[:256]
    
    heldout_rst = compute_correlations(x_heldout, y_heldout, dist, model)
    
    with open(f'./results/{args.task}_{args.net}_heldout_{args.seed}.npy', 'wb') as f:
        np.save(f, heldout_rst)
    

class Runner:
    """
        The interface of landscape/model/explorer is compatible with FLEXS benchmark.
        - Fitness Landscape EXploration Sandbox (FLEXS)
          https://github.com/samsinai/FLEXS
    """
    
    def __init__(self, args):
        self.num_rounds = args.num_rounds
        self.num_queries_per_round = args.num_queries_per_round
        self.method = args.alg
        self.use_wandb = args.use_wandb
        self.init_model = args.init_model
        self.args = args

    def run(self, landscape, starting_sequence, model, explorer, starting_dataset=None):
        self.results = pd.DataFrame()
        if starting_dataset is None:
            starting_fitness = landscape.get_fitness([starting_sequence])[0]
            self.update_results(0, [starting_sequence], [starting_fitness])
        else:
            self.update_results(0, starting_dataset[0].tolist(), starting_dataset[1].tolist())
        
        if self.method in ['gfn-al', 'gfn_seq_editor']: #update dataset too
            explorer.dataset = BioSeqDataset(explorer.args, explorer.tokenizer, init_data=starting_dataset)
        
        for round in range(1, self.num_rounds+1):
            round_start_time = time.time()
            
            if self.args.use_rank_based_proxy_training:
                # model.train(self.sequence_buffer, self.fitness_buffer)
                model.train_prioritized(self.sequence_buffer, self.fitness_buffer, init_model=self.init_model)
            elif self.method is not 'gfn_seq_editor':
                model.train(self.sequence_buffer, self.fitness_buffer)
                
            # if round == 1:
            #     # TODO: evaluate proxy model
            #     evaluate_proxy(model, self.sequence_buffer, self.fitness_buffer, self.args)
            #     break
                
            sequences, model_scores = explorer.propose_sequences(self.results)
            assert len(sequences) <= self.num_queries_per_round
            true_scores = landscape.get_fitness(sequences)

            round_running_time = time.time()-round_start_time
            self.update_results(round, sequences, true_scores, starting_sequence, round_running_time)
            # self.update_results(round, sequences, true_scores, starting_sequence, round_running_time)
            
            if self.method == 'gfn-al': #update dataset too
                explorer.dataset.add((sequences, true_scores))
                explorer.round = round
              
        if self.use_wandb:
            # Convert pandas DataFrame to a wandb.Table and log it
            wandb_table = wandb.Table(dataframe=self.results)
            wandb.log({"rst": wandb_table})
        self.results.to_csv(f'./results/{self.args.task}/{self.method}_{self.args.name}_{self.args.seed}.csv', index=False)
    
    def update_results(self, round, sequences, true_scores, wt=None, running_time=0.0):
        self.results = self.results.append(
            pd.DataFrame({
                "round": round,
                "sequence": sequences,
                "true_score": true_scores
            })
        )
        print('round: {}  max fitness score: {:.3f}  running time: {:.2f} (sec)'.format(round, self.results['true_score'].max(), running_time))
        if self.use_wandb:
            top100 = self.results.nlargest(self.args.num_queries_per_round, 'true_score')
            div100 = mean_pairwise_distances(top100['sequence'])
            avg100 = top100['true_score'].mean()
            initial = self.results[self.results['round']==0]['sequence']
            novelty100 = mean_novelty(top100['sequence'], initial)
            median = np.percentile(top100['true_score'], 50)
            dist100_from_wt = mean_distances_from_wt(top100['sequence'], wt) if wt is not None else 0
            queried_dist100 = mean_distances_from_wt(sequences, wt) if wt is not None else 0
            queried_div = mean_pairwise_distances(sequences)
            queried_score = np.mean(true_scores)
            wandb.log({'round': round,  'running_time': running_time,
                       'max_fitness': self.results['true_score'].max(), 'median_topK': median, 'avg_topK': avg100,
                       'div100': div100, 'novelty100': novelty100, 'dist100_from_wt': dist100_from_wt,
                       'queried_div': queried_div, 'queried_score': queried_score, 'queried_dist_from_wt': queried_dist100})
    
    @property
    def sequence_buffer(self):
        return self.results['sequence'].to_numpy()

    @property
    def fitness_buffer(self):
        return self.results['true_score'].to_numpy()

