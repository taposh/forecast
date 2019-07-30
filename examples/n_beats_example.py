from kpforecast.ml import GenericNBeatsBlock

forward_length = 3
backwards_length = 6

GenericNBeatsBlock(f_b_dim=(forward_length, backwards_length),
                   layer_dim=4,
                   thetas_dim=(6,10),
                   shared_g_theta=False)
