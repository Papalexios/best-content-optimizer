import { GeneratedContent, SiteInfo, ExpandedGeoTargeting } from './index.tsx';

export type WpConfig = {
    url: string;
    username: string;
};

// =================================================================
// 💎 PREMIUM SCHEMA.ORG MARKUP GENERATOR
// Optimized for AI Overviews, SGE, and SERP Dominance
// =================================================================

const ORGANIZATION_NAME = "Your Company Name";
const DEFAULT_AUTHOR_NAME = "Expert Author";

/**
 * Creates a Person schema with rich E-E-A-T signals
 */
function createPersonSchema(siteInfo: SiteInfo, primaryKeyword: string) {
    return {
        "@type": "Person",
        "@id": `${siteInfo.authorUrl}#person`,
        "name": siteInfo.authorName || DEFAULT_AUTHOR_NAME,
        "url": siteInfo.authorUrl || undefined,
        "sameAs": siteInfo.authorSameAs && siteInfo.authorSameAs.length > 0 ? siteInfo.authorSameAs : undefined,
        "description": `Expert content creator specializing in ${primaryKeyword}`,
        "knowsAbout": [primaryKeyword],
    };
}

/**
 * Creates Organization schema with enhanced credibility signals
 */
function createOrganizationSchema(siteInfo: SiteInfo, wpConfig: WpConfig) {
    return {
        "@type": "Organization",
        "@id": `${wpConfig.url}#organization`,
        "name": siteInfo.orgName || ORGANIZATION_NAME,
        "url": siteInfo.orgUrl || wpConfig.url,
        "logo": siteInfo.logoUrl ? {
            "@type": "ImageObject",
            "@id": `${wpConfig.url}#logo`,
            "url": siteInfo.logoUrl,
            "width": 600,
            "height": 60,
        } : undefined,
        "sameAs": siteInfo.orgSameAs && siteInfo.orgSameAs.length > 0 ? siteInfo.orgSameAs : undefined,
    };
}

/**
 * Creates LocalBusiness schema for geo-targeted content
 */
function createLocalBusinessSchema(siteInfo: SiteInfo, geoTargeting: ExpandedGeoTargeting, wpConfig: WpConfig) {
    return {
        "@type": "LocalBusiness",
        "@id": `${wpConfig.url}#localbusiness`,
        "name": siteInfo.orgName || ORGANIZATION_NAME,
        "url": siteInfo.orgUrl || wpConfig.url,
        "address": {
            "@type": "PostalAddress",
            "addressLocality": geoTargeting.location,
            "addressRegion": geoTargeting.region,
            "postalCode": geoTargeting.postalCode,
            "addressCountry": geoTargeting.country,
        },
        "geo": {
            "@type": "GeoCoordinates",
            "addressCountry": geoTargeting.country
        }
    };
}

/**
 * Creates premium NewsArticle schema (better than Article for rankings)
 */
function createArticleSchema(
    content: GeneratedContent,
    wpConfig: WpConfig,
    orgSchema: any,
    personSchema: any,
    articleUrl: string
) {
    const today = new Date().toISOString();
    const textContent = content.content.replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ').trim();
    const wordCount = textContent.split(/\s+/).filter(Boolean).length;
    const readingTimeMinutes = Math.ceil(wordCount / 200);
    
    // Extract all H2 headings for article structure
    const headings = [...content.content.matchAll(/<h2[^>]*>(.*?)<\/h2>/g)].map(m => m[1]);

    const articleSchema: any = {
        "@type": "NewsArticle",
        "@id": `${articleUrl}#article`,
        "mainEntityOfPage": {
            "@type": "WebPage",
            "@id": articleUrl
        },
        "headline": content.title,
        "description": content.metaDescription,
        "image": content.imageDetails
            .filter(img => img.generatedImageSrc)
            .map(img => ({
                "@type": "ImageObject",
                "url": img.generatedImageSrc,
                "caption": img.altText
            })),
        "datePublished": today,
        "dateModified": today,
        "author": personSchema,
        "publisher": orgSchema,
        "keywords": [content.primaryKeyword, ...content.semanticKeywords].join(", "),
        "articleSection": content.primaryKeyword,
        "wordCount": wordCount,
        "timeRequired": `PT${readingTimeMinutes}M`,
        "inLanguage": "en-US",
        "isAccessibleForFree": true,
        "speakable": {
            "@type": "SpeakableSpecification",
            "cssSelector": ["h1", "h2", "h3"]
        },
    };

    // Add article sections for better structure
    if (headings.length > 0) {
        articleSchema.hasPart = headings.map((heading, index) => ({
            "@type": "WebPageElement",
            "@id": `${articleUrl}#section-${index + 1}`,
            "name": heading,
        }));
    }

    return articleSchema;
}

/**
 * Creates WebSite schema with SearchAction for AI crawlers
 */
function createWebSiteSchema(wpConfig: WpConfig, orgSchema: any) {
    return {
        "@type": "WebSite",
        "@id": `${wpConfig.url}#website`,
        "url": wpConfig.url,
        "name": orgSchema.name,
        "publisher": {
            "@id": `${wpConfig.url}#organization`
        },
        "potentialAction": {
            "@type": "SearchAction",
            "target": {
                "@type": "EntryPoint",
                "urlTemplate": `${wpConfig.url}/?s={search_term_string}`
            },
            "query-input": "required name=search_term_string"
        }
    };
}

/**
 * Creates BreadcrumbList for enhanced navigation and SERP display
 */
function createBreadcrumbSchema(content: GeneratedContent, wpConfig: WpConfig, articleUrl: string) {
    return {
        "@type": "BreadcrumbList",
        "@id": `${articleUrl}#breadcrumb`,
        "itemListElement": [
            {
                "@type": "ListItem",
                "position": 1,
                "name": "Home",
                "item": wpConfig.url
            },
            {
                "@type": "ListItem",
                "position": 2,
                "name": content.primaryKeyword,
                "item": `${wpConfig.url}/category/${content.primaryKeyword.toLowerCase().replace(/\s+/g, '-')}`
            },
            {
                "@type": "ListItem",
                "position": 3,
                "name": content.title,
                "item": articleUrl
            }
        ]
    };
}

/**
 * Creates FAQPage schema from structured data
 */
function createFaqSchema(faqData: { question: string, answer: string }[]) {
    if (!faqData || faqData.length === 0) return null;
    
    const mainEntity = faqData
        .filter(faq => faq.question && faq.answer)
        .map(faq => ({
            "@type": "Question",
            "name": faq.question,
            "acceptedAnswer": {
                "@type": "Answer",
                "text": faq.answer,
            },
        }));

    if (mainEntity.length === 0) return null;

    return {
        "@type": "FAQPage",
        "mainEntity": mainEntity,
    };
}

/**
 * Creates HowTo schema for instructional content
 */
function createHowToSchema(content: GeneratedContent, articleUrl: string) {
    const headings = [...content.content.matchAll(/<h2[^>]*>(.*?)<\/h2>/g)].map(m => m[1]);
    
    // Check if content is instructional
    const hasSteps = content.content.match(/<ol>[\s\S]*?<\/ol>/) || 
                     headings.some(h => h.toLowerCase().includes('step') || 
                                       h.toLowerCase().includes('how to'));
    
    if (!hasSteps || headings.length < 3) return null;

    const textContent = content.content.replace(/<[^>]+>/g, ' ').trim();
    const wordCount = textContent.split(/\s+/).filter(Boolean).length;
    const readingTimeMinutes = Math.ceil(wordCount / 200);

    return {
        "@type": "HowTo",
        "@id": `${articleUrl}#howto`,
        "name": content.title,
        "description": content.metaDescription,
        "totalTime": `PT${readingTimeMinutes}M`,
        "step": headings.slice(0, 8).map((heading, index) => ({
            "@type": "HowToStep",
            "position": index + 1,
            "name": heading,
            "text": heading,
            "url": `${articleUrl}#section-${index + 1}`
        }))
    };
}

/**
 * Creates VideoObject schemas for embedded YouTube videos
 */
function createVideoObjectSchemas(content: GeneratedContent, articleUrl: string) {
    const schemas: any[] = [];
    const videoMatches = [...content.content.matchAll(/youtube\.com\/embed\/([^"?]+)/g)];
    
    videoMatches.forEach((match, index) => {
        const videoId = match[1];
        schemas.push({
            "@type": "VideoObject",
            "@id": `${articleUrl}#video-${index + 1}`,
            "name": `Video: ${content.title} - Part ${index + 1}`,
            "description": content.metaDescription,
            "thumbnailUrl": `https://i.ytimg.com/vi/${videoId}/maxresdefault.jpg`,
            "uploadDate": new Date().toISOString(),
            "contentUrl": `https://www.youtube.com/watch?v=${videoId}`,
            "embedUrl": `https://www.youtube.com/embed/${videoId}`,
            "inLanguage": "en-US"
        });
    });

    return schemas.length > 0 ? schemas : null;
}

/**
 * Main schema generation function - creates comprehensive @graph
 */
export function generateFullSchema(
    content: GeneratedContent,
    wpConfig: WpConfig,
    siteInfo: SiteInfo,
    faqData?: { question: string, answer: string }[],
    geoTargeting?: ExpandedGeoTargeting
): object {
    const articleUrl = `${wpConfig.url.replace(/\/+$/, '')}/${content.slug}`;
    const schemas: any[] = [];
    
    // 1. Organization (Publisher)
    const organizationSchema = createOrganizationSchema(siteInfo, wpConfig);
    schemas.push(organizationSchema);
    
    // 2. Person (Author)
    const personSchema = createPersonSchema(siteInfo, content.primaryKeyword);
    schemas.push(personSchema);
    
    // 3. WebSite
    const websiteSchema = createWebSiteSchema(wpConfig, organizationSchema);
    schemas.push(websiteSchema);
    
    // 4. NewsArticle (Primary Content)
    const articleSchema = createArticleSchema(content, wpConfig, organizationSchema, personSchema, articleUrl);
    
    // Add geo-targeting if enabled
    if (geoTargeting?.enabled && geoTargeting.location) {
        articleSchema.contentLocation = {
            "@type": "Place",
            "name": geoTargeting.location,
            "address": {
                "@type": "PostalAddress",
                "addressLocality": geoTargeting.location,
                "addressRegion": geoTargeting.region,
                "addressCountry": geoTargeting.country,
                "postalCode": geoTargeting.postalCode
            }
        };
        articleSchema.spatialCoverage = {
            "@type": "Place",
            "name": geoTargeting.location
        };
        
        // Add LocalBusiness schema for geo-targeted content
        const localBusinessSchema = createLocalBusinessSchema(siteInfo, geoTargeting, wpConfig);
        schemas.push(localBusinessSchema);
    }
    
    schemas.push(articleSchema);
    
    // 5. BreadcrumbList
    const breadcrumbSchema = createBreadcrumbSchema(content, wpConfig, articleUrl);
    schemas.push(breadcrumbSchema);
    
    // 6. FAQPage
    if (faqData && faqData.length > 0) {
        const faqSchema = createFaqSchema(faqData);
        if (faqSchema) schemas.push(faqSchema);
    }
    
    // 7. HowTo (if applicable)
    const howToSchema = createHowToSchema(content, articleUrl);
    if (howToSchema) schemas.push(howToSchema);
    
    // 8. VideoObject schemas
    const videoSchemas = createVideoObjectSchemas(content, articleUrl);
    if (videoSchemas) schemas.push(...videoSchemas);

    return {
        "@context": "https://schema.org",
        "@graph": schemas,
    };
}

/**
 * Wraps schema in proper script tags for WordPress
 */
export function generateSchemaMarkup(schemaObject: object): string {
    if (!schemaObject || !Object.prototype.hasOwnProperty.call(schemaObject, '@graph') || 
        (schemaObject as any)['@graph'].length === 0) {
        return '';
    }
    
    const schemaScript = `<script type="application/ld+json">\n${JSON.stringify(schemaObject, null, 2)}\n</script>`;
    
    // ════════════════════════════════════════════════════════════════════════════════
    // CRITICAL: WordPress REST API strips <script> tags by default for security.
    // The ONLY reliable method is wrapping in Gutenberg's Custom HTML block.
    // This preserves the schema and prevents it from being stripped or displayed as text.
    // ════════════════════════════════════════════════════════════════════════════════
    return `\n\n<!-- wp:html -->\n${schemaScript}\n<!-- /wp:html -->\n\n`;
}
