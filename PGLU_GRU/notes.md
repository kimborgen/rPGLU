The sglu, renamed to PGLU, demonstrated the power of short-term modelling trough neuron potential gating signals, but it lacks the 
longer term modelling ability of LSTM and thus falls short 1-5% on modelling benchmarks. This model, a recurrent Potential-Gated and Long-Term Memory model shall aim to rectify this by adding some form of long-term memory to the PGLU. 

1 layer (exp 6)
+------------------+------------+---------------+
|     Modules      | Parameters | Requires grad |
+------------------+------------+---------------+
|   pglu1.tresh    |     64     |      True     |
| pglu1.decay_rate |     64     |      True     |
| pglu1.W_i.weight |    640     |      True     |
|  pglu1.W_i.bias  |     64     |      True     |
| pglu1.W_r.weight |    8192    |      True     |
|  pglu1.W_r.bias  |     64     |      True     |
| pglu1.W_z.weight |    8192    |      True     |
|  pglu1.W_z.bias  |     64     |      True     |
| pglu1.W_c.weight |    8192    |      True     |
|  pglu1.W_c.bias  |     64     |      True     |
|  fc_out.weight   |    576     |      True     |
|   fc_out.bias    |     9      |      True     |
+------------------+------------+---------------+
Total Params: 26185
Total Grad Params: 26185


2 layer (exp 5)
+------------------+------------+---------------+
|     Modules      | Parameters | Requires grad |
+------------------+------------+---------------+
|   pglu1.tresh    |     64     |      True     |
| pglu1.decay_rate |     64     |      True     |
| pglu1.W_i.weight |    640     |      True     |
|  pglu1.W_i.bias  |     64     |      True     |
| pglu1.W_r.weight |    8192    |      True     |
|  pglu1.W_r.bias  |     64     |      True     |
| pglu1.W_z.weight |    8192    |      True     |
|  pglu1.W_z.bias  |     64     |      True     |
| pglu1.W_c.weight |    8192    |      True     |
|  pglu1.W_c.bias  |     64     |      True     |
|   pglu2.tresh    |     64     |      True     |
| pglu2.decay_rate |     64     |      True     |
| pglu2.W_i.weight |    4096    |      True     |
|  pglu2.W_i.bias  |     64     |      True     |
| pglu2.W_r.weight |    8192    |      True     |
|  pglu2.W_r.bias  |     64     |      True     |
| pglu2.W_z.weight |    8192    |      True     |
|  pglu2.W_z.bias  |     64     |      True     |
| pglu2.W_c.weight |    8192    |      True     |
|  pglu2.W_c.bias  |     64     |      True     |
|  fc_out.weight   |    576     |      True     |
|   fc_out.bias    |     9      |      True     |
+------------------+------------+---------------+
Total Params: 55241
Total Grad Params: 55241