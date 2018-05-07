

%------------------------------------------------------------
%    Author: Behnam Anjomruz
%    Email: behnam.anjomruz@gmail.com
%------------------------------------------------------------

clear all;

% loads an image
im_clean = im2double(imread('5.jpg'));

% adds Poisson noise
scale=1.2;
Imnoisy = (1/scale)*imnoise(im_clean*scale,'poisson');
% Imnoisy=imresize(Imnoisy1,[1800 1800]);

%caclulating MaxSTD of all learning images

IMGsDir = dir('*.jpg');

nImgs = length(IMGsDir);
% stds=(nImgs);
for i=1:nImgs
   currentfilename = IMGsDir(i).name;
   currentimage = imresize(imread(currentfilename),[1800 1800]);
   images{i}=im2double(currentimage);
   stds{i} = std2(images{i});
end

stds2Mat=cell2mat(stds);
stdMax=max(stds2Mat(:));
NoNNFs = ceil(stdMax/2);


% [tempx tempy]=size(Imnoisy);
% for s=1:nImgs
%     [xh yw]=size (images{s});
%     if tempx<xh
%         tempx=xh;
%     end
%     if tempy<yw
%         tempy=yw;
%     end
% end

for l=1:NoNNFs
    stdNNFs=10*NoNNFs;
    net{l}=newff(minmax(Imnoisy),[9 1],{'tansig' 'purelin'},'trainscg');
        
end




for i=1:nImgs
   currentIm=images{i};
   [x y]=size(currentIm);
   j=3;
   k=3;
   while j<x-2
       tempIm=currentIm(j-2:j+2,k-2:k+2);
       T=zeros(size(tempIm));
       if((j==x-2) && (k ~= y-2))
           k=k+1;
           j=3;
       end
       if((k==y-1) && (j==x-1))
           break;
       end
       
       tmpStd=std2(tempIm);
       
       %%%%%%%%%%%%%%
       
       for l=1:NoNNFs
           tempnet=net{l};
           tempIm=zeros(size(tempIm));
           if(0 <= tmpStd < ((stdNNFs)/(10^l)))
                net{l}=train(tempnet,tempIm');
           end
       end
       j=j+1;
       %%%%%%%%%%%%%%
   end
end

% im=im2double(imread('5.jpg'));
[xh yw]=size(Imnoisy);
% cropedIm=im2double(Imnoisy(2:1788, 2:1788));
% for l=1:NoNNFs
%     tempnet=net{l};
%     y=tempnet(cropedIm);
%     perf=perform(tempnet,cropedIm,y);
% end


model = {};
model.weightsSig = 3;
% denoising stride
model.step = 1;

% denoise
fprintf('starting to denoise...\n');
tstart = tic;
im_denoised = fdenoiseNeural(Imnoisy, 10, model);
telapsed = toc(tstart);
fprintf('Done! Loading the weights and denoising took %.1f seconds\n',telapsed);

% PSNR values
psnr_noisy = getPSNR(Imnoisy, im_clean, 255);
psnr_denoised = getPSNR(im_denoised, im_clean, 255);
fprintf('PSNR: noisy: %.2fdB, denoised: %.2fdB\n',psnr_noisy,psnr_denoised);

subplot(131); imagesc(im_clean); s = gca;           title('clean'); axis image
subplot(132); imagesc(Imnoisy, get(s, 'CLim'));    title('noisy'); axis image
subplot(133); imagesc(im_denoised, get(s, 'CLim')); title('denoised'); axis image
colormap(gray);



