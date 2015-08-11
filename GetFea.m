clear all;
fea1=[];
ii = 1;
for k=1:32
    query1=['../Project/faceClustering/alignedFaces/' num2str(k) '/*.JPG'];
    query2=['../Project/faceClustering/alignedFaces/' num2str(k) '/*.jpg'];
    ax = [dir(query1); dir(query2)];
    len1 = length(ax);
    
    for i=1:len1
        path1 = ['/home/jess/Project/faceClustering/alignedFaces/' num2str(k) '/' ax(i).name];
        
        fileID = fopen('examples/_temp/file_list.txt','w');
        fprintf(fileID,'%s 0\n', path1);
        fclose(fileID);
        % Save library paths
        MatlabPath = getenv('LD_LIBRARY_PATH');
        % Make Matlab use system libraries
        setenv('LD_LIBRARY_PATH',getenv('PATH'))
        if (exist('examples/_temp/fea/', 'dir'))
            rmdir('examples/_temp/fea', 's');
        end
        
        disp(['Extract feature for ' num2str(i) 'th face at ' num2str(k) 'th cluster.']);
        isPass = system('python python/googlenet_test.py -i ../examples/_temp/file_list.txt -o fea1.txt');
        if isPass == 0
            [A] = textread('fea1.txt', '%f\n');
        else
            disp('Failed to execute caffe code');
        end
        % Reassign old library paths
        setenv('LD_LIBRARY_PATH',MatlabPath)
        
        fea1(:, ii) = single(A);
        ii = ii + 1;
        break;
    end
    breaLINk;
end

save('deepFeaFace.mat');