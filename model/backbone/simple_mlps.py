import torch.nn as nn

def return_simple_mlps(options, feat_dim):
    """
    Returns a simple MLPs based on the options.

    Args:
        options (str): Options for the simple MLPs.
            'deep': Deep MLPs.
            'half': Half MLPs.
            'CMpp': CMpp MLPs.
            'CMpp_half': CMpp Half MLPs.
        
        feat_dim (int): Feature dimension.

    Returns:
        nn.Sequential: A simple MLPs.
    """
    if options == 'deep':
        channel_dim_of_shape_feats = feat_dim
        shape_mlp = nn.Sequential(nn.Conv1d((feat_dim//3) * 3, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 )
    

    elif options == 'half':
        channel_dim_of_shape_feats = feat_dim//2
        shape_mlp = nn.Sequential(nn.Conv1d((feat_dim//3) * 3, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 )
    
    
    elif options == 'CMpp':
        channel_dim_of_shape_feats = feat_dim
        shape_mlp = nn.Sequential(nn.Conv1d((feat_dim//3) * 3, feat_dim, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 )
    
    elif options == 'CMpp_half':
        channel_dim_of_shape_feats = feat_dim
        shape_mlp = nn.Sequential(nn.Conv1d((feat_dim//3) * 3, feat_dim//2, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim//2),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim//2, feat_dim, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 nn.Conv1d(feat_dim, feat_dim, kernel_size=1, bias=False),
                                 nn.InstanceNorm1d(feat_dim),
                                 nn.LeakyReLU(negative_slope=0.2),
                                 )
    

    else:
        raise ValueError(f"options must be in ['deep', 'half', 'CMpp', 'CMpp_half'], but got {options}")
    
    return shape_mlp, channel_dim_of_shape_feats