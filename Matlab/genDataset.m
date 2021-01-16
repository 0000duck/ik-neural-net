%% Initialization

disp('Program started');

dof=6; %degrees of freedom
theta=zeros(1,dof); %joint angles
jh=zeros(1,dof); %CoppeliaSim joint handles
iterations=1000; %number of code iterations

inFile='..\Dataset\UR10\joint_values.xlsx'; % input file
outFile='..\Dataset\UR10\ee_pos_ori.xlsx'; % output file

% Do the connection with CoppeliaSim
% sim=remApi('remoteApi','extApi.h'); % using the header (requires a compiler)
sim=remApi('remoteApi'); % using the prototype file (remoteApiProto.m)
sim.simxFinish(-1); % just in case, close all opened connections
clientID=sim.simxStart('127.0.0.1',19999,true,true,5000,5);

%% Main program

if (clientID>-1)
    disp('Connected to remote API server');
    
    % Retreive handles from CoppeliaSim
    for i = 1 : dof
        [~,jh(i)]=sim.simxGetObjectHandle(clientID, strcat('UR10_joint', int2str(i)) , sim.simx_opmode_blocking);
    end
    [~,tip]=sim.simxGetObjectHandle(clientID, 'tip' , sim.simx_opmode_blocking);
    
    for it = 1 : iterations
        
        fprintf('Iteration number: %d ____________________________________________\n', it);
        
        % Set random joint values 
        for j = 1 : dof
            theta(j)=randi([-360 360],1,1);
        end

        % Send joint values to CoppeliaSim
        disp('Target joint value in degrees');
        disp(theta);
        for k = 1 : dof
            sim.simxSetJointTargetPosition(clientID, jh(k), deg2rad(theta(k)), sim.simx_opmode_streaming);
        end
        
        % Get final position and orientation and print them
        [~,pos] = sim.simxGetObjectPosition(clientID,tip,-1,sim.simx_opmode_blocking);
        [~,ori] = sim.simxGetObjectOrientation(clientID,tip,-1,sim.simx_opmode_blocking);
        ee_pos_ori=[pos rad2deg(ori)];
        
        disp('Final position and orientation (xe, ye, ze, alpha, beta, gamma)');
        disp(ee_pos_ori);
        
        writematrix(theta,inFile,'WriteMode','append');
        writematrix(ee_pos_ori,outFile,'WriteMode','append');
        pause(1);
    end
    
    
else
    disp('Failed connecting to remote API server');
end
sim.delete(); % call the destructor!

clear;

disp('Program ended');