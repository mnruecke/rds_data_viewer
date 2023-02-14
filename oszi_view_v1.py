
###################################################################################
# Messung 13.2.2023 - iron oxide 250 nm
# 
# 8x Mittelungen Lecroy x 8 Mittelungen manual
# folder: # ./


###################################################################################
# Dataset description


main_title = "old RDS 50 kHz - Iron Oxide 250 nm"

data_dir = "./rds13022023_iron_oxide_250b/"    

file_type = 'TRC' # Tektronix: 'CSV'; Lecroy: 'TRC' 


DELETE_EXISTING_PLOTS = True  
harmonic_plot_list = range(1,17)#[2,3,4,5,10,11,16,17,22,23]                                                            


CREATE_NPY_TEMP_DATA  = False
FORCE_RAW_DATA_RELOAD = True # ignore preprocessed data in temp_dir

import platform
import os
if CREATE_NPY_TEMP_DATA: # -> *.npy loads faster than *.csv

    # Directory for preprocessed data:
    temp_dir_name = "rds_50kHz_IronOxide_x8x8/"    
    
    if platform.system() == 'Windows':
        temp_dir_root = "C:/Users/marti/Downloads/"
    elif platform.system() == 'Linux':
        temp_dir_root = "/home/martin/Downloads/"
    temp_dir = temp_dir_root + temp_dir_name   
   
    if not os.path.exists( temp_dir ):
        os.mkdir( temp_dir )

# file collection: [1...N]
# blocks in file collection: [1..M][M+1..K]..[..N]
# interleaved data sets in block: [1a,1b,1c ... Ma, Mb, Mc...]
NUM_OF_FILES = 16
NUM_OF_DATA_BLOCKS = 2
NUM_OF_FILES_IN_DATA_BLOCK = 8
NUM_OF_INTERLEAVED_DATASETS = 4 # Datasets in a block

sig_labels_block_1 = [
                        'Iron Oxide 200 nm (1a)','Leer (1a)',
                        'Iron Oxide 200 nm (1b)','Leer (1b)',
                     ] 

sig_labels_block_2 = [
                        'Iron Oxide 200 nm (2a)','Leer (2a)',
                        'Iron Oxide 200 nm (2b)','Leer (2b)',
                     ] 

sig_labels = (
               sig_labels_block_1 + sig_labels_block_2
             )

# Deselect data with '0'
plot_option_1 = [
                        [ 1, 1, 1, 1, ], # Block 1
                        [ 1, 1, 1, 1, ], # Block 2
                        
                    ]

# No baseline
plot_option_2 = [
                        [ 1, 0, 1, 0, ], # Block 1
                        [ 1, 0, 1, 0, ], # Block 2
                        
                    ]


visibility_matrix = plot_option_1

# Data set order for plotting 
data_ids = list(range(len(sig_labels_block_1)))
# sample_x, ref_x, sample_y, ref_y, ...
START_IDX_SAMPLE = 0 # even #: sample
START_IDX_REF    = 1 # odd  #: reference


# Center frequency f_cen = (f_chx + f_chy)/2 used for down mixing
F_CHX = 53250 # Hz
F_CHY = 53350 # Hz

BANDWIDTH = 10000 # Bandwidth around the downmixed harmonic in Hz

SAMPLES_PER_SEC = 25e6 # sampling rate of data files
DOWN_SAMPLING   = 1
SAMPLING_TIME_IN_SEC = 0.10 # 1 / SAMPLES_PER_SEC * N_SAMPLES

# Plotting options
PLOT_FULL_BANDWIDTH_SIGNAL = 1
PLOT_HARMONICS = 1 # Downmixed signal

PLOT_REFERENCE = 1


###################################################################################
# Function definitions for data processing

import numpy as np
if file_type == 'CSV':
    from numpy import genfromtxt
elif file_type == 'TRC':
    from trc_reader import Trc
    trc = Trc()
import matplotlib.pyplot as plt
import matplotlib


def get_full_sized_window():
    try:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
    except:
        print( "'%matplotlib qt' not selected")


def signal_down_mixing( signal, n_harmonic_shift, bandwidth,
                        real_and_imag = False ):
    
    signal_ft = np.fft.fft( signal )

    df          = bandwidth # Hz
    f_center    = (F_CHX + F_CHY) / 2
    f_shift     = n_harmonic_shift * f_center
    
    df_pts       = int( df       * SAMPLING_TIME_IN_SEC )
    f_shift_pts  = int( f_shift  * SAMPLING_TIME_IN_SEC )
    
    y_ft_lhs = 2* np.roll( signal_ft, -f_shift_pts )
    y_ft_lhs = y_ft_lhs[ : df_pts]
    y_ft_lhs = y_ft_lhs + 0 * y_ft_lhs
    
    amp_corr = float(len( y_ft_lhs )) / float(len( signal_ft ))
    sig_complex = amp_corr * np.fft.ifft( y_ft_lhs )
    
    
    sig_real = np.transpose( np.real( sig_complex ) )
    sig_imag = np.transpose( np.imag( sig_complex ) )
    
    if real_and_imag:
        return sig_real, sig_imag
    else:
        return sig_real


def downsampling_data( t, y, decimation_factor ):

    # Excess data points: for TEK: 0, for Lecroy: 2, 
    if file_type == 'CSV':
        dismiss_first_data_points = 0 
    elif file_type == 'TRC':
        dismiss_first_data_points = 2
    y = y[ dismiss_first_data_points :: ]
    t = t[ dismiss_first_data_points :: ]
     
    if decimation_factor == 1:
        return t, y   
    
    y = np.reshape( y, [ len( y ) // decimation_factor, decimation_factor ] )
    y = np.sum( y, 1 ) / decimation_factor  
    t = t[ :: decimation_factor ]   
    
    return t, y

def read_data_file( data_directory, file_number, file_type_ ):
    ''' Read *.csv or *.trc files '''
    settings = []
    fName = ""

    if file_type_ == 'CSV':
        fName = (
                 data_dir + "TEK"
                 + str( file_number ).rjust(5,'0') + ".CSV"
                 )
        data = genfromtxt( fName, delimiter=',')[15::]
        t, y = data[:,0], data[:,1]
    
    elif file_type_ == 'TRC':
        fName = (
                 data_dir + "C1--trace--" 
                 + str( file_number ).rjust(5,'0') + ".trc"
                 )
        t, y, settings = trc.open(fName)           

    return t, y, fName


def average_dataset( data_set_start, last_data_file, num_of_interleaved_data_sets ):
    
    sig_avg = []
    n_avg = 0
    for file_i in range( data_set_start, last_data_file, num_of_interleaved_data_sets ):

        
        t_full, y_full, fName = read_data_file(
                                                data_dir,
                                                file_i,
                                                file_type
                                               )
        
        t, y = downsampling_data( t_full, y_full, DOWN_SAMPLING )
        
        if len(sig_avg) == 0:
            sig_avg = np.zeros(len(y))
        sig_avg = sig_avg + y
        print( fName + "     data set start: " + str( data_set_start ).rjust(3,' '))
        n_avg = n_avg + 1
    sig_avg = sig_avg / n_avg
    
    return t, sig_avg, n_avg


def read_data_sets( range_start, range_end, num_of_interleaved_data_sets ):
    
    sig_container = []
    for data_set in range( num_of_interleaved_data_sets ):
        t, sig_, n = average_dataset( range_start + data_set, range_end,
                                          num_of_interleaved_data_sets )
        sig_container.append(sig_)    
        
    return t, sig_container, n

###############################################################################
# Data processing

try:
    # Skip reading raw data files if data is already in temp_dir
    if not FORCE_RAW_DATA_RELOAD:
        if os.path.exists( temp_dir + 'sig.npy' ):
            sig = np.load( temp_dir + 'sig.npy')
            t   = np.load( temp_dir + 't.npy')
            n   = np.load( temp_dir + 'n_avg.npy')
            print( "Get processed data from 'sig.npy' in: " )
            print( temp_dir )
            
        else:
            print( "'sig.npy' not found in:" )
            print( temp_dir )
            print( "read raw-files instead" )
        
    len(sig) != 0 
    
except:              
    sig = []
    for block_i in range( NUM_OF_DATA_BLOCKS ):     
        
        t, sig_, n = read_data_sets(
                block_i    * NUM_OF_FILES_IN_DATA_BLOCK, 
               (block_i+1) * NUM_OF_FILES_IN_DATA_BLOCK,
                NUM_OF_INTERLEAVED_DATASETS
            )
        
        if len(sig) == 0:
            sig = sig_
        else:
            sig = np.append( sig, sig_, axis=0 )
          
    if CREATE_NPY_TEMP_DATA and os.path.exists( temp_dir ):         
        np.save( temp_dir + 'sig.npy', sig)
        np.save( temp_dir + 'n_avg.npy', n)
        np.save( temp_dir + 't.npy', t)
    
    sig_ = []

###################################################################################
# Plotting..

if DELETE_EXISTING_PLOTS:
    plt.close('all')

if platform.system() == 'Windows':    
    matplotlib.rcParams.update({'font.size': 22})
elif platform.system() == 'Linux':    
    matplotlib.rcParams.update({'font.size': 12})
    # (command works differently for 'tk' vs 'qt')


time_unit  = 1000 # s / ms

sample_ids = data_ids[ START_IDX_SAMPLE ::2 ]

ref_ids    = data_ids[ START_IDX_REF ::2 ]
ref_x_list = ref_ids
ref_y_list = ref_x_list[-1:] + ref_x_list[:-1]     

assert( NUM_OF_FILES == NUM_OF_DATA_BLOCKS * NUM_OF_FILES_IN_DATA_BLOCK )
assert( np.size( sig_labels ) == NUM_OF_INTERLEAVED_DATASETS * NUM_OF_DATA_BLOCKS )    
assert( np.size( visibility_matrix ) == np.size( sig_labels ))
 
    
if PLOT_FULL_BANDWIDTH_SIGNAL:  
    
    plt.figure()
    get_full_sized_window()
    
    plt.title( main_title + "\n" + 
               "Differenzsignale: " + str(n) + " Mittelungen;"
               + " MS/s: " +  str( SAMPLES_PER_SEC / DOWN_SAMPLING / 1e6 ))  

    for block_i in range( NUM_OF_DATA_BLOCKS ):    
        
        file_offset = block_i * NUM_OF_INTERLEAVED_DATASETS
        
        for dat_i in sample_ids:       
            
            if visibility_matrix [block_i] [dat_i] == 0:
                continue       
            
            t_ = t * time_unit
            
            trace_ = (   sig[data_ids[ dat_i   ] +file_offset ]
                       - sig[data_ids[ dat_i+1 ] +file_offset ]
                       )
            
            label_ = (    sig_labels[data_ids[ dat_i   ] +file_offset ] + ' - '
                        + sig_labels[data_ids[ dat_i+1 ] +file_offset ]
                        )
            
            plt.plot( t_ , trace_, label = label_ )       
        
    if PLOT_REFERENCE:
        
        for block_i in range( NUM_OF_DATA_BLOCKS ):
            
            file_offset = block_i * NUM_OF_INTERLEAVED_DATASETS
            
            for ref_x_i, ref_y_i in zip( ref_x_list, ref_y_list ): 
                
                if visibility_matrix [block_i] [ref_x_i] == 0:
                    continue     
                
                t_ = t * time_unit
                
                trace_ = (   sig[data_ids[ ref_x_i ] +file_offset ]
                           - sig[data_ids[ ref_y_i ] +file_offset ]
                           )
                
                label_ = (   sig_labels[data_ids[ ref_x_i ] +file_offset ] + ' - '
                           + sig_labels[data_ids[ ref_y_i ] +file_offset ]                         
                           )
                
                plt.plot( t_, trace_, label = label_ )

 
    plt.xlabel('t[ms]')
    plt.ylabel('Amplitude[V]')
    plt.legend( loc='upper right' )  
        
    
        
if PLOT_HARMONICS:
    # Compare harmonics among the data sets
    

    for harmonic in harmonic_plot_list:
    
        plt.figure()
        get_full_sized_window()
        
        plt.title(   str( harmonic ) + "th Harmonic for: " + main_title )
            
        for block_i in range( NUM_OF_DATA_BLOCKS ):
            
            file_offset = block_i * NUM_OF_INTERLEAVED_DATASETS   
        
            for dat_i in sample_ids:
                
                if visibility_matrix [block_i] [dat_i] == 0:
                    continue  
                    
                t_ = t * time_unit
                
                raw_trace_ = sig[ dat_i +file_offset ] - sig[ dat_i+1 +file_offset ]
                trace_ = signal_down_mixing( raw_trace_, harmonic, BANDWIDTH )
                
                t_downmixed = t_ [ :: len(t)//len(trace_) ]
                
                label_ = (   sig_labels[data_ids[ dat_i   ] +file_offset ] + ' - '
                           + sig_labels[data_ids[ dat_i+1 ] +file_offset ]
                           )
                                          
                plt.plot( t_downmixed, trace_, label = label_ )  
    
    
        if PLOT_REFERENCE:
            
            for block_i in range( NUM_OF_DATA_BLOCKS ):    
                
                file_offset = block_i * NUM_OF_INTERLEAVED_DATASETS   
            
                for ref_x_i, ref_y_i in zip( ref_x_list, ref_y_list ):
                    
                    if visibility_matrix [block_i] [ref_x_i] == 0:
                        continue        
                    
                    t_ = t * time_unit
                    
                    raw_trace_ = sig[ ref_x_i +file_offset ] - sig[ ref_y_i +file_offset ]
                    trace_ = signal_down_mixing( raw_trace_, harmonic, BANDWIDTH )
                    
                    t_downmixed = t_ [ :: len(t)//len(trace_) ]     
                    
                    label_ = (   sig_labels[data_ids[ ref_x_i ] +file_offset ] + ' - '
                               + sig_labels[data_ids[ ref_y_i ] +file_offset ]                         
                               )           
                    
                    plt.plot( t_downmixed, trace_, label = label_ )  
        
        raw_trace = []
        trace_ = []
            
        plt.legend( loc='upper right' )  
        plt.xlabel('t[ms]')
        plt.ylabel('Amplitude[V]')    
        