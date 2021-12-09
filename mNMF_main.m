%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% multichannel NMF over pairwise mixes of audio dataset 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters for all inputs 
fs_resample = 16000;
num_sources = 2; 
fft_len = 4096; 
shift_len = 2048; 
num_bases = 20;% ~ 10x number of sources
max_iter = 300; 

% Ranom seed
seed = 1; 
RandStream.setGlobalStream(RandStream('mt19937ar','Seed',seed));


% metric output: [SAR SIR SDR]
eval_out = [];

folder_dir = uigetdir + "/instrument_samples";
files = dir(fullfile(folder_dir,'*.wav'));
for k = 3 %1:length(files)
  file_path = strcat('./instrument_samples/', files(k).name);
  clear sig1;
  [sig1(:,:), fs] = audioread(file_path);
  for j = 4%1:length(files)
      if ~ strcmp(files(j).name, files(k).name)
          file_path = strcat('./instrument_samples/', files(j).name);
          clear sig2;
          [sig2(:,:), fs] = audioread(file_path);
          
          % truncate to length of shortest signal 
          % mix two audio signals at equal gains
          clear sig;
          len = min(size(sig1,1), size(sig2, 1));
          sig(:,:,1) = rescale(sig1(1:len,:), -0.5, 0.5);
          sig(:,:,2) = rescale(sig2(1:len,:), -0.5, 0.5);
          
          % downsample to 16kHz 
          clear sig_resample;
          sig_resample(:,:,1) = resample(sig(:,:,1), fs_resample, fs, 100); 
          sig_resample(:,:,2) = resample(sig(:,:,2), fs_resample, fs, 100);
          
          % mix input audio clips
          clear input;
          input(:,1) = sig_resample(:,1,1) + sig_resample(:,1,2);
          input(:,2) = sig_resample(:,2,1) + sig_resample(:,2,2);
%           if abs(max(max(input))) > 1.00 
%               error('input clipped.\n');
%           end
          
          % calculate metrics for input signals 
          clear source;
          source(:,1) = sig_resample(:,1,1);
          source(:,2) = sig_resample(:,1,2);
          
          clear input_metric;
          input_metric(1,1) = 10.*log10( sum(sum(squeeze(sig_resample(:,1,1)).^2)) ./ sum(sum(squeeze(sig_resample(:,2,1)).^2)) );
          input_metric(2,1) = 10.*log10( sum(sum(squeeze(sig_resample(:,2,1)).^2)) ./ sum(sum(squeeze(sig_resample(:,1,1)).^2)) );

          
          % run mNMF algorithm 
          [output,cost] = mNMF(input, num_bases, fft_len, shift_len, max_iter);
          
          % calculate SDR, SIR, and SAR using library 
          [SDR,SIR,SAR] = bss_eval_sources(squeeze(output(:,1,:)).', source.');
          SDRimp = SDR - input_metric
          SIRimp = SIR - input_metric
          SAR
          
          eval_out = [eval_out; SAR SIRimp SDRimp];
          
          in1_name = strsplit(files(k).name, '.');
          in1_name = in1_name{1};
          in2_name = strsplit(files(j).name, '.');
          in2_name = in2_name{1};
          dir_name = strcat('./instrument_output/', in1_name, '_', in2_name);
          mkdir(dir_name);
          
          audiowrite(sprintf('%s/mix.wav', dir_name), input, fs_resample); 
          audiowrite(sprintf('%s/in1.wav', dir_name), sig_resample(:,:,1), fs_resample);
          audiowrite(sprintf('%s/in2.wav', dir_name), sig_resample(:,:,2), fs_resample); 
          audiowrite(sprintf('%s/out1.wav', dir_name), output(:,:,1), fs_resample); 
          audiowrite(sprintf('%s/out2.wav', dir_name), output(:,:,2), fs_resample);
          
          % save plot of cost function
          figure;
          semilogy( (0:max_iter), cost );
          xlabel('Number of Iterations');
          ylabel('Cost Function Value');
          saveas(gcf, strcat('./instrument_output/',in1_name, '_', in2_name, '_cost.m'));
          
          
      end 
     
  end
  
end

% save evaluation metrics for all runs to csv in output folder 
writematrix(eval_out, strcat('./instrument_output/', 'bss_eval.csv'));