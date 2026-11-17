# Results: graphs and tables

To rebuild the graphs and `eval_scores_*.csv` from the recorded logs, run `.venv\Scripts\python results\make_plots.py`.

| file | what it shows |
|---|---|
| `learning_curve_configA_5M.png` | DQN config A, seed 17: evaluation score vs training decisions, 0–5M. Mean line with min–max shading, a 0 line, and a marker at 2M where the run was resumed. The dip at 2.1M is unexplained. |
| `old_vs_new_settings.png` | Old-settings pilot vs config A over the first 1M decisions. Config A reaches about −9 by 1M; the old settings reached about −17 by 700k. |
| `eval_scores_configA_5M.csv` | The table behind the learning curve: mean, min and max score and the number of games at every evaluation step. |
| `eval_scores_oldsettings_pilot.csv` | The same table for the old-settings pilot (0–700k). |
| `metrics_table.csv` | Behaviour metrics from `experiments/04_diagnosis/compare_metrics.py`: score, bottom-third paddle share, ball tracking, action gap. The only row so far is the 600k old-settings baseline. |
| `gpu_throughput.json` | GPU vs CPU speed (decisions/s), RAM by replay size and time estimates from the throughput test. |
| `behaviour/<model>/summary.txt` | `demos/explain_ai.py` text summary for one model: scores, where the paddle sits, ball tracking, how early it "sees" a lost point. |
| `behaviour/<model>/summary.json` | The same, machine-readable. |
| `behaviour/<model>/1_paddle_position_hist.png` | Where the paddle sits, both while the ball approaches and when no ball is in play. |
| `behaviour/<model>/2_action_frequency.png` | How often each of the 6 actions is chosen. |
| `behaviour/<model>/3_max_q_over_time.png` | The agent's value estimate over the game, with points won and lost marked. |
| `behaviour/<model>/4_paddle_vs_ball_y.png` | Paddle height vs ball height over time: does it follow the ball? |
| `behaviour/<model>/5_incoming_ball_reach.png` | For each incoming ball, where the paddle started vs where the ball arrived (returned or missed). |

The `behaviour/` folders are `dqn_oldsettings_seed17_0.6M` (camps at the bottom, lost 1:21), `dqn_configA_seed17_2.0M` (lost 14:21) and `dqn_configA_seed17_5.0M` (won 21:11). Each covers one held-out evaluation game.

The submission notebook's own results (the original 100k CPU runs) are in `Assignment2/results/`. Don't move them.
