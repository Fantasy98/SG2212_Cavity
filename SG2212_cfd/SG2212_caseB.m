% Navier-Stokes solver,
% adapted for course SG2212
% KTH Mechanics
%
% Depends on avg.m and DD.m
%
% Code version: 
% 20180222

clear all

%------------------------------------------

%  test for Case A for now/ 7 Mar

lid_driven_cavity=1;

if (lid_driven_cavity==1)
  % Parameters for test case I: Lid-driven cavity
  % The Richardson number is zero, i.e. passive scalar.
  
  Pr = 0.71;     % Prandtl number
  Re = 200;      % Reynolds number
  Ri = 0.;       % Richardson number
  
  dt = 0.001;      % time step
  Tf = 20;       % final time
  Lx = 1;        % width of box
  Ly = 1;       % height of box
  Nx = 50;      % number of cells in x
  Ny = 50;      % number of cells in y
  ig = 20;      % number of iterations between output
  
  % Boundary and initial conditions:
  Utop = 1.;   Ubottom = 0.;
  % IF TEMPERATURE: Tbottom = 1.; Ttop = 0.;
  % IF TEMPERATURE: namp = 0.;
end

%-----------------------------------------

% Number of iterations
Nit = 30e3;
% Spatial grid: Location of corners 
x = linspace(0,1,Nx+1); 
y = linspace(0,1,Ny+1); 
% Grid spacing

dx =  Lx/Nx;
dy = Ly/Ny;
% Boundary conditions: 
uN = x*0+Utop;    vN = avg(x,2)*0;
uS = x*0+Ubottom;          vS = avg(x,2)*0;
uW = avg(y,2)*0;          vW = y*0;
uE = avg(y,2)*0;          vE = y*0;
tN = x.*0;          tS = y.*0;
% Initial conditions , contain the inner points 
U = zeros(Nx-1,Ny);
V = zeros(Nx,Ny-1);
% linear profile for T with random noise
% IF TEMPERATURE: T = ... + namp*rand(Nx,Ny)
% Time series
tser = [];
Tser = [];

%-----------------------------------------

% Compute system matrices for pressure 
% First set homogeneous Neumann condition all around
% Laplace operator on cell centres: Fxx + Fyy
Lp = kron(speye(Ny), DD(Nx,dx)) + kron( DD(Ny,dy),speye(Nx) );
% Set one Dirichlet value to fix pressure in that point
Lp(1,:) = 0 ; Lp(1,1) = 1;

% Here you can pre-compute the LU decomposition
% [LLp,ULp] = lu(Lp);
%-----------------------------------------

% Progress bar (do not replace the ... )
fprintf(...
    '[         |         |         |         |         ]\n')


%-----------------------------------------

% Main loop over iterations

for k = 1:Nit
     
   % include all boundary points for u and v (linear extrapolation
   % for ghost cells) into extended array (Ue,Ve)
   Ue = [uW ; U ; uE];
   Ue = [2.*uS'-Ue(:,1) Ue 2.*uN'-Ue(:,end)];

   Ve = [vS' V  vN']; 
   Ve =[ 2.*vW-Ve(1,:); Ve ; 2.*vE-Ve(end,:)];
%    Ve = [2.*vS'-V(:,1) V 2.*vN'-V(:,end)];
%    Ve = [vW ; Ve ; vE]; 

   % averaged (Ua,Va) of u and v on corners 
   % For example U_(i+1/2,j+1/2)  should be extraplated along y-axis then 
   % U_(i+1/2,j+1/2)  = avg(U_(i+1/2,j),U_(i+1/2,j+1)) 
   Ua = avg(Ue,2);
   Va = avg(Ve,1);
 
   % construct individual parts of nonlinear terms
   % diff(A,x,y) A= target matrix , x=order of derivative , y = along  axis
   dUVdx = (1/dx) .* diff( Ua(:,2:end-1).*Va(:,2:end-1),1,1);  
   dUVdy = (1/dy) .* diff( Ua(2:end-1,:).*Va(2:end-1,:),1,2); 
%    dU2dx = (1/dx) .* diff(Ua(:,2:end-1).*Ua(:,2:end-1)  ,1,1);
%    dV2dy = (1/dy) .* diff( Va(2:end-1,:).*Va(2:end-1,:) ,1,2); 
%    dUVdx = (1/dx) .* diff( Ua.*Va,1,1);  
%    dUVdy = (1/dy) .* diff( Ua.*Va ,1,2); 
U2 = avg(Ue.*Ue,1);
V2 = avg(Ve.*Ve,2);
   dU2dx = (1/dx) .* diff(U2(:,2:end-1),1,1);
   dV2dy = (1/dy) .* diff(V2(2:end-1,:),1,2); 
   % treat viscosity explicitly
   viscu = diff(Ue(:,2:end-1),2,1)./(dx^2) + diff(Ue(2:end-1,:),2,2)./(dy^2); 
   viscv = diff(Ve(:,2:end-1),2,1)./(dx^2) + diff(Ve(2:end-1,:),2,2)./(dy^2); 
   
   % buoyancy term
   % IF TEMPERATURE: fy = ...
         
   % compose final nonlinear term + explicit viscous terms
   U = U + (dt/Re) .* viscu - dt.*( dU2dx + dUVdy);
   V = V + (dt/Re) .* viscv - dt.*( dV2dy + dUVdx) ; % IF TEMPERATURE: dt*fy;
   
   % pressure correction, Dirichlet P=0 at (1,1)
   rhs = ( diff([uW;U;uE],1,1)/dx + diff([vS' V  vN'],1,2)/dy )/dt;
   rhs = reshape(rhs,Nx*Ny,1);
   rhs(1) = 0; 
   P = Lp\rhs;
   % alternatively, you can use the pre-computed LU decompositon
   % P = ...;
   % or gmres
   % P = gmres(Lp, rhs, [], tol, maxit);
   % or as another alternative you can use GS / SOR from homework 6
	% [PP, r] = GS_SOR(omega, Nx, Ny, hx, hy, L, f, p0, tol, maxit);
   P = reshape(P,Nx,Ny);
   
   % apply pressure correction   
   U = U - dt* diff(P,1,1)./dx;
   V = V - dt* diff(P,1,2)./dy;
   
   % Temperature equation
   % IF TEMPERATURE: Te = ... 
   % IF TEMPERATURE: Tu = ...
   % IF TEMPERATURE: Tv = ...
   % IF TEMPERATURE: H = ...
   % IF TEMPERATURE: T = T + dt*H;
   
   %-----------------------------------------
   
   % progress bar
   if floor(51*k/Nit)>floor(51*(k-1)/Nit), fprintf('.'), end
   
   % plot solution if needed
   if k==1|floor(k/ig)==k/ig

    %%%%compute divergence on cell centres
%      if (1==1)
%        div = diff([uW;U;uE])/dx + diff([vS' V vN'],1,2)/dy;
% 
%        figure(1);clf; hold on;
%        contourf(avg(x,2),avg(y,2),div');colorbar
%        axis equal; axis([0 Lx 0 Ly]);
%        title(sprintf('divergence at t=%g',k*dt))
%        drawnow
%      end 
     
     % compute velocity on cell corners
   Ue = [uW ; U ; uE];
   Ue = [2.*uS'-Ue(:,1) Ue 2.*uN'-Ue(:,end)];

   Ve = [vS' V  vN']; 
   Ve =[ 2.*vW-Ve(1,:); Ve ; 2.*vE-Ve(end,:)];
   
   
     Ua = avg(Ue,2);
     Va = avg(Ve,1);
     Len = sqrt(Ua.^2 + Va.^2+eps);

%      figure(2);clf;hold on;
%     % contourf(avg(x,2),avg(y,2),P');colorbar
%      contourf(x,y,sqrt(Ua.^2+Va.^2)',20,'k-');colorbar
%      quiver(x,y,(Ua./Len)',(Va./Len)',.4,'k-')
%      axis equal; axis([0 Lx 0 Ly]);
%      title(sprintf('u at t=%g',k*dt))
%      drawnow
     
     % IF TEMPERATURE: % compute temperature on cell corners
     % IF TEMPERATURE: Ta = ...
     
     % IF TEMPERATURE: figure(3); clf; hold on;
     % IF TEMPERATURE: contourf(x,y,Ta',20,'k-');colorbar
     % IF TEMPERATURE: quiver(x,y,(Ua./Len)',(Va./Len)',.4,'k-')
     % IF TEMPERATURE: axis equal; axis([0 Lx 0 Ly]);
     % IF TEMPERATURE: title(sprintf('T at t=%g',k*dt))
     % IF TEMPERATURE: drawnow
     
    % Time history
%      if (1==1)
%        figure(4); hold on;
%        tser = [tser k*dt];
%        Tser = [Tser Ue(ceil((Nx+1)/2),ceil((Ny+1)/2))];
%        plot(tser,abs(Tser),Linewidth=2)
%        yline(0.2,'-.');xline(Re./10,'-.b','linewidth',2)
%        title(sprintf('Probe signal at x=%g, y=%g',...
%              x(ceil((Nx+1)/2)),y(ceil((Ny+1)/2))))
%        set(gca,'yscale','log')
%        xlabel('time t');ylabel('u(t)')
%        legend('Probe','U=0.2','time=Re/10')
%      end
   end
end
     
fprintf('\n')
%%
figure(10)
LEQ= sqrt(Ua.^2 + Va.^2)
M=csvread('Re200N50Y0.5.csv',1,0);
Ux=M(:,1);Uy=M(:,2);Uz=M(:,3);xa=M(:,6);ya=M(:,7);
Ubar=sqrt(Ux.^2 + Uy.^2);
plot(x,LEQ(:,26),xa,Ubar,Linewidth=2);
legend('Matlab','OpenFoam')
xlabel('location');ylabel('Speed');title('Speed Profile at ly=0.5'); 
set(gca,'fontsize',16);
%%
figure(11)
LEQ= sqrt(Ua.^2 + Va.^2)
M=csvread('Re200N50X0.5.csv',1,0);
Ux=M(:,1);Uy=M(:,2);Uz=M(:,3);xa=M(:,6);ya=M(:,7);
Ubar=sqrt(Ux.^2 + Uy.^2);
plot(x,LEQ(Ny/2,:),xa,Ubar,Linewidth=2);
legend('Matlab','OpenFoam')
xlabel('location');ylabel('Speed');title('Speed Profile U at lx=0.5'); 
set(gca,'fontsize',16);

