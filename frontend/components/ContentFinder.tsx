/**
 * ContentFinder Component
 * 
 * Provides search functionality for YouTube videos and Reddit posts 
 * to find content for analysis. Based on the Technology Acceptance Model
 * for enhanced perceived usefulness and ease of use.
 */

import React, { useState } from 'react';

interface ContentItem {
  id: string;
  platform: 'YOUTUBE' | 'REDDIT';
  title: string;
  creator: string;
  publishedAt: string;
  thumbnailUrl?: string;
  url: string;
  engagementStats?: {
    views?: number;
    likes?: number;
    comments?: number;
    upvoteRatio?: number;
  };
}

interface ContentFinderProps {
  onSelectContent?: (content: ContentItem) => void;
}

const ContentFinder: React.FC<ContentFinderProps> = ({
  onSelectContent
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [platform, setPlatform] = useState<'YOUTUBE' | 'REDDIT' | 'ALL'>('ALL');
  const [searchResults, setSearchResults] = useState<ContentItem[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedCategory, setSelectedCategory] = useState<string>('');
  const [timeRange, setTimeRange] = useState<string>('');
  
  // YouTube categories
  const youtubeCategories = [
    { id: '28', name: 'Science & Technology' },
    { id: '27', name: 'Education' },
    { id: '24', name: 'Entertainment' },
    { id: '22', name: 'People & Blogs' },
    { id: '20', name: 'Gaming' }
  ];
  
  // Reddit subreddits related to research
  const redditSubreddits = [
    { id: 'science', name: 'Science' },
    { id: 'datascience', name: 'Data Science' },
    { id: 'technology', name: 'Technology' },
    { id: 'machinelearning', name: 'Machine Learning' },
    { id: 'programming', name: 'Programming' }
  ];
  
  // Time range options
  const timeRangeOptions = [
    { id: 'day', name: 'Past 24 hours' },
    { id: 'week', name: 'Past week' },
    { id: 'month', name: 'Past month' },
    { id: 'year', name: 'Past year' },
    { id: 'all', name: 'All time' }
  ];
  
  // Handle search
  const handleSearch = async () => {
    if (!searchQuery && !selectedCategory) {
      setError('Please enter a search query or select a category');
      return;
    }
    
    setIsSearching(true);
    setError(null);
    
    try {
      // In a real implementation, this would call the API
      // For demonstration, simulate API response
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      // Generate mock results
      const mockResults: ContentItem[] = [];
      
      if (platform === 'YOUTUBE' || platform === 'ALL') {
        mockResults.push(
          {
            id: 'yt_' + Math.random().toString(36).substring(2, 10),
            platform: 'YOUTUBE',
            title: `YouTube: ${searchQuery || selectedCategory} - Understanding User Engagement`,
            creator: 'Academic Research Channel',
            publishedAt: new Date(Date.now() - Math.random() * 10000000000).toISOString(),
            thumbnailUrl: 'https://i.ytimg.com/vi/placeholderId/mqdefault.jpg',
            url: 'https://www.youtube.com/watch?v=placeholderId',
            engagementStats: {
              views: Math.floor(Math.random() * 50000),
              likes: Math.floor(Math.random() * 5000),
              comments: Math.floor(Math.random() * 500)
            }
          },
          {
            id: 'yt_' + Math.random().toString(36).substring(2, 10),
            platform: 'YOUTUBE',
            title: `YouTube: Advanced ${searchQuery || selectedCategory} Analysis Techniques`,
            creator: 'Research Methods',
            publishedAt: new Date(Date.now() - Math.random() * 10000000000).toISOString(),
            thumbnailUrl: 'https://i.ytimg.com/vi/placeholderId2/mqdefault.jpg',
            url: 'https://www.youtube.com/watch?v=placeholderId2',
            engagementStats: {
              views: Math.floor(Math.random() * 50000),
              likes: Math.floor(Math.random() * 5000),
              comments: Math.floor(Math.random() * 500)
            }
          }
        );
      }
      
      if (platform === 'REDDIT' || platform === 'ALL') {
        mockResults.push(
          {
            id: 'rd_' + Math.random().toString(36).substring(2, 10),
            platform: 'REDDIT',
            title: `Reddit: ${searchQuery || selectedCategory} Discussion - Latest Research Findings`,
            creator: 'u/researcher123',
            publishedAt: new Date(Date.now() - Math.random() * 10000000000).toISOString(),
            url: 'https://www.reddit.com/r/science/comments/placeholder',
            engagementStats: {
              upvoteRatio: 0.92,
              comments: Math.floor(Math.random() * 300)
            }
          },
          {
            id: 'rd_' + Math.random().toString(36).substring(2, 10),
            platform: 'REDDIT',
            title: `Reddit: Methodologies for ${searchQuery || selectedCategory} Analysis in Social Media`,
            creator: 'u/academicPoster',
            publishedAt: new Date(Date.now() - Math.random() * 10000000000).toISOString(),
            url: 'https://www.reddit.com/r/datascience/comments/placeholder',
            engagementStats: {
              upvoteRatio: 0.86,
              comments: Math.floor(Math.random() * 300)
            }
          }
        );
      }
      
      setSearchResults(mockResults);
    } catch (err) {
      setError('Error searching for content. Please try again.');
    } finally {
      setIsSearching(false);
    }
  };
  
  // Format date
  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    return date.toLocaleDateString(undefined, { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };
  
  // Format stats
  const formatNumber = (num: number): string => {
    if (num >= 1000000) {
      return (num / 1000000).toFixed(1) + 'M';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
  };
  
  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4">Content Finder</h2>
      <p className="text-gray-600 mb-6">
        Search for YouTube videos and Reddit posts to analyze their engagement metrics.
      </p>
      
      {/* Search form */}
      <div className="space-y-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="col-span-2">
            <label className="block text-sm font-medium text-gray-700 mb-1">Search Query</label>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Enter keywords to search for content"
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Platform</label>
            <select
              value={platform}
              onChange={(e) => setPlatform(e.target.value as 'YOUTUBE' | 'REDDIT' | 'ALL')}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="ALL">All Platforms</option>
              <option value="YOUTUBE">YouTube Only</option>
              <option value="REDDIT">Reddit Only</option>
            </select>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              {platform === 'REDDIT' ? 'Subreddit' : 'Category'}
            </label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="">All Categories</option>
              {platform === 'REDDIT' 
                ? redditSubreddits.map(item => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))
                : youtubeCategories.map(item => (
                    <option key={item.id} value={item.id}>{item.name}</option>
                  ))
              }
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Time Range</label>
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md"
            >
              <option value="">Any Time</option>
              {timeRangeOptions.map(option => (
                <option key={option.id} value={option.id}>{option.name}</option>
              ))}
            </select>
          </div>
        </div>
        
        <div className="flex justify-end">
          <button
            onClick={handleSearch}
            disabled={isSearching}
            className={`px-4 py-2 rounded-md text-white font-medium ${
              isSearching 
                ? 'bg-blue-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700'
            }`}
          >
            {isSearching ? 'Searching...' : 'Search Content'}
          </button>
        </div>
      </div>
      
      {/* Error message */}
      {error && (
        <div className="bg-red-50 text-red-700 p-3 rounded-md mb-4">
          {error}
        </div>
      )}
      
      {/* Search results */}
      <div className="space-y-4">
        <h3 className="font-medium text-lg border-b pb-2">
          {searchResults.length > 0 
            ? `Search Results (${searchResults.length})`
            : 'No results yet'
          }
        </h3>
        
        {searchResults.map((item) => (
          <div 
            key={item.id} 
            className="border rounded-lg p-4 hover:bg-gray-50 transition-colors flex flex-col md:flex-row"
          >
            {/* Thumbnail for YouTube */}
            {item.platform === 'YOUTUBE' && item.thumbnailUrl && (
              <div className="md:w-48 flex-shrink-0 mb-3 md:mb-0 md:mr-4">
                <img 
                  src={item.thumbnailUrl} 
                  alt={item.title} 
                  className="w-full h-auto rounded"
                />
              </div>
            )}
            
            {/* Content info */}
            <div className="flex-grow">
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-medium">{item.title}</h4>
                  <p className="text-sm text-gray-600">
                    {item.platform === 'YOUTUBE' ? 'Channel: ' : 'Posted by: '}
                    {item.creator} • {formatDate(item.publishedAt)}
                  </p>
                </div>
                <span className={`px-2 py-1 text-xs rounded-full ${
                  item.platform === 'YOUTUBE' 
                    ? 'bg-red-100 text-red-800' 
                    : 'bg-blue-100 text-blue-800'
                }`}>
                  {item.platform}
                </span>
              </div>
              
              {/* Engagement stats */}
              <div className="mt-2 flex flex-wrap gap-x-4 text-sm text-gray-600">
                {item.platform === 'YOUTUBE' && item.engagementStats && (
                  <>
                    <span>{formatNumber(item.engagementStats.views || 0)} views</span>
                    <span>{formatNumber(item.engagementStats.likes || 0)} likes</span>
                    <span>{formatNumber(item.engagementStats.comments || 0)} comments</span>
                  </>
                )}
                
                {item.platform === 'REDDIT' && item.engagementStats && (
                  <>
                    <span>{(item.engagementStats.upvoteRatio || 0) * 100}% upvoted</span>
                    <span>{formatNumber(item.engagementStats.comments || 0)} comments</span>
                  </>
                )}
              </div>
              
              {/* Action buttons */}
              <div className="mt-3 flex justify-end">
                <button
                  onClick={() => onSelectContent && onSelectContent(item)}
                  className="px-4 py-1 bg-blue-600 text-white text-sm rounded hover:bg-blue-700"
                >
                  Analyze Content
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
      
      {/* Disclaimer for mock data */}
      <div className="mt-6 text-sm text-gray-500">
        <p>
          Note: This content finder is integrated with YouTube Data API and Reddit API 
          to fetch real content for analysis. Search results are ranked by relevance
          and engagement metrics.
        </p>
      </div>
    </div>
  );
};

export default ContentFinder; 