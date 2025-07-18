from option_pricer import OptionType, OptionStyle, BarrierType
from price import price_option

call_price, stddev = price_option(
    option_type=OptionType.EUROPEAN,
    barrier_type=BarrierType.NONE,
    style=OptionStyle.CALL,
    S=100, K=100, r=0.05, sigma=0.2, T=1.0,
    steps=250,         
    paths=1_000_000    
)

print(f"European Call Price: {call_price:.4f} ± {stddev:.4f}")

call_price, stddev = price_option(
    option_type=OptionType.ASIAN,
    barrier_type=BarrierType.NONE,
    style=OptionStyle.CALL,
    S=100,        
    K=100,        
    r=0.05,       
    sigma=0.2,    
    T=1.0,        
    steps=250,    
    paths=1_000_000
)

print(f"Asian Call Price: {call_price:.4f} ± {stddev:.4f}")

call_price, stddev = price_option(
    option_type=OptionType.BARRIER,
    barrier_type=BarrierType.UP_AND_OUT,
    style=OptionStyle.CALL,
    barrier=120,
    S=100,        
    K=100,        
    r=0.05,       
    sigma=0.2,    
    T=1.0,        
    steps=250,    
    paths=1_000_000  
)

print(f"Up and Out Call Price: {call_price:.4f} ± {stddev:.4f}")
