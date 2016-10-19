clc
clear all
close all
imaqreset
%% creating arduino and servo object
% a = arduino('com10', 'uno', 'Libraries', 'Servo');
%   s1 = servo(a, 'D9')
%   s2 = servo(a, 'D10')
  
%% creating video object
vid=videoinput('winvideo',1,'YUY2_640x480');
set(vid,'ReturnedColorSpace','rgb');
set(vid,'FramesPerTrigger',1);
set(vid,'TriggerRepeat',inf);
triggerconfig(vid,'manual'); 
start(vid);
pause(2);
trigger(vid);
im=getdata(vid);
%% Create a cascade detector object and face detector.
faceDetector = vision.CascadeObjectDetector();
bbox = step(faceDetector,im);
% Draw the returned bounding box around the detected face.
im = insertShape(im, 'Rectangle', bbox);
figure; imshow(im); title('Detected face');
% Convert the first box into a list of 4 points
% This is needed to be able to visualize the rotation of the object.
bboxPoints = bbox2points(bbox(1, :));
%% Identify Facial Features To Track
% Detect feature points in the face region.
points = detectMinEigenFeatures(rgb2gray(im), 'ROI', bbox);

% Display the detected points.
figure, imshow(im), hold on, title('Detected features');
plot(points);
%% Initialize a Tracker to Track the Points

% Create a point tracker and enable the bidirectional error constraint to
% make it more robust in the presence of noise and clutter.
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);

% Initialize the tracker with the initial point locations and the initial
% video frame.
points = points.Location;
initialize(pointTracker, points, im);
%% Initialize a Video Player to Display the Results
% Create a video player object for displaying video frames.
videoPlayer  = vision.VideoPlayer('Position',...
    [100 100 [size(im, 2), size(im, 1)]+30]);
%% track the face
% Make a copy of the points to be used for computing the geometric
% transformation between the points in the previous and the current frames
oldPoints = points;

for i=1:100
    trigger(vid);
    im=getdata(vid);
    
    % Track the points. Note that some points may be lost.
    [points, isFound] = step(pointTracker, im);
    visiblePoints = points(isFound, :);
    oldInliers = oldPoints(isFound, :);
    if size(visiblePoints, 1) >= 2 % need at least 2 points    
    % Estimate the geometric transformation between the old points
    % and the new points and eliminate outliers
     [xform, oldInliers, visiblePoints] = estimateGeometricTransform(...
            oldInliers, visiblePoints, 'similarity', 'MaxDistance', 4);
        
        % Apply the transformation to the bounding box points
        bboxPoints = transformPointsForward(xform, bboxPoints);
        [myX myY]=mycentroid(bboxPoints)
        % Insert a bounding box around the object being tracked
        bboxPolygon = reshape(bboxPoints', 1, []);
        im = insertShape(im, 'Polygon', bboxPolygon, ...
            'LineWidth', 2);
                
        % Display tracked points
        im = insertMarker(im, visiblePoints, '+', ...
            'Color', 'red');  
        im =insertMarker(im,[myX,myY],'*','Color','blue');
        
        % Reset the points
        oldPoints = visiblePoints;
        setPoints(pointTracker, oldPoints);        
    end
    
    % Display the annotated video frame using the video player object
    step(videoPlayer, im);
    
end
% Clean up
release(videoPlayer);
release(pointTracker);
 stop(vid);
 delete(vid);
 %clear(vid);
imaqreset
    